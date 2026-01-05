import torch.nn.functional as F
import random
import json
import pickle
import torch
from pathlib import Path
from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"



def candidate_filter(candidates,
            num_candidates=1,
            token_to_flip=None,
            adv_passage_ids=None,
            ppl_model=None, 
            ppl_tokenizer=None,
            model_tokenizer=None,
            device=device):
    """
    并行化版本：无需新函数，直接在内部利用 Tensor 操作批量计算。
    """
    # -------------------------------------------------------------------------
    # 1. 快速构建 Batch (替换掉 For 循环)
    # -------------------------------------------------------------------------
    # candidates: [num_cand] (例如 100 个候选词的 ID)
    num_cands = len(candidates)
    
    # 复制原始 Trigger N 份 -> [num_cand, seq_len]
    # 比如: [1, 5] -> [100, 5]
    # a, b .repeat(x, y) = ax, by
    temp_adv_passages = adv_passage_ids.repeat(num_cands, 1)
    
    # 一次性把所有候选词填入对应的位置
    # 这一步相当于原来的: temp_adv_passage[:, token_to_flip] = candidate
    # token to flip is the same
    temp_adv_passages[:, token_to_flip] = candidates

    # -------------------------------------------------------------------------
    # 2. 批量计算 PPL (替换掉 compute_perplexity)
    # -------------------------------------------------------------------------
    with torch.no_grad():
        # 直接把 100 个句子一起扔进去
        # 注意：这里不需要传 labels，因为传了 labels 模型会返回平均 loss，我们需要每个句子的 loss
        outputs = ppl_model(temp_adv_passages)
        logits = outputs.logits  # Shape: [num_cand, seq_len, vocab_size]
        # output is the probability distribution of every token in the entire sequence

        # 准备 Labels (就是输入本身)
        labels = temp_adv_passages
        
        # 手动计算 CrossEntropyLoss，设置 reduction='none' 以保留每个样本的 Loss
        # View 操作: [N, Seq, Vocab] -> [N*Seq, Vocab]
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        # 计算所有 Token 的 Loss
        # PyTorch 的 CrossEntropy 需要 (N, C) 和 (N)
        token_losses = loss_fct(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1)
        )
        
        # 变回 [num_cand, seq_len] 的形状
        token_losses = token_losses.view(num_cands, -1)
        
        # 计算每个句子的平均 Loss (按行求和再除以长度)
        sentence_loss = token_losses.mean(dim=1)
        
        # PPL = exp(loss)
        # 我们这里要取反 (-PPL)，因为后面是用 topk 选最大的，而我们希望 PPL 越小越好
        ppl_scores = -torch.exp(sentence_loss)

    # -------------------------------------------------------------------------
    # 3. 择优录取
    # -------------------------------------------------------------------------
    # 选出分数最高的（其实是 PPL 最小的）前 K 个
    _, top_k_ids = ppl_scores.topk(num_candidates)
    
    # 返回对应的候选词 ID
    return candidates[top_k_ids]








def load_db_ad_simple(database_samples_dir="data/finetune/data_samples_train.json", 
                      db_dir="data/memory", 
                      model_code="None", 
                      model=None, 
                      tokenizer=None, 
                      device='cuda', 
                      split_ratio=1.0): # 1. 新增参数

    # 为了保证文件名唯一，将 split_ratio 加入缓存文件名中
    # 例如: embeddings_bert_0.1.pkl
    cache_filename = f"{db_dir}/embeddings_{model_code}_ratio_{split_ratio}.pkl"

    # 检查是否存在对应采样率的缓存文件
    if Path(cache_filename).exists():
        print(f"Loading cached embeddings from {cache_filename}...")
        with open(cache_filename, "rb") as f:
            embeddings = pickle.load(f)
        
        # 处理不同模型加载后的 tensor 形状问题
        if isinstance(embeddings, list):
             # 如果是 list (DPR/BGE等逻辑中原本是存list)，转为 tensor
            embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
            db_embeddings = embeddings.squeeze(1)
        else:
            # 如果是 stack 后的 tensor
            embeddings = embeddings.to(device)
            db_embeddings = embeddings.squeeze(1)
            
        return db_embeddings

    # 如果没有缓存，则加载数据并重新计算
    print(f"No cache found. Loading data from {database_samples_dir} with ratio {split_ratio}...")
    
    with open(database_samples_dir, "rb") as f:
        full_samples = json.load(f)

    # 2. & 3. 核心修改：使用随机采样代替 [:20000] 切片
    if split_ratio < 1.0:
        database_samples = []
        # 设置随机种子可确保每次采样的"岛屿"形状一致（可选）
        random.seed(42) 
        for sample in full_samples:
            if random.random() < split_ratio:
                database_samples.append(sample)
        print(f"Sampled {len(database_samples)} items from {len(full_samples)} total.")
    else:
        database_samples = full_samples

    embeddings = []

    # 下面是原本的 embedding 计算逻辑，保持不变，适配不同模型
    if 'contrastive' in model_code or 'classification' in model_code or 'bert' in model_code:
        for sample in tqdm(database_samples):
            ego = sample["ego"]
            perception = sample["perception"]
            prompt = f"{ego} {perception}"
            tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)
                if 'bert' in model_code:
                    query_embedding = model(input_ids, attention_mask).pooler_output
                else:
                    query_embedding = model(input_ids, attention_mask)
                embeddings.append(query_embedding)
        
        # Stack tensor
        embeddings = torch.stack(embeddings, dim=0).to(device)

    elif 'dpr' in model_code or 'bge' in model_code:
        for sample in tqdm(database_samples):
            ego = sample["ego"]
            perception = sample["perception"]
            prompt = f"{ego} {perception}"
            tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)
                query_embedding = model(input_ids, attention_mask).pooler_output
                
                # DPR/BGE 在原代码中是存 list
                if 'dpr' in model_code:
                     query_embedding = query_embedding.detach().cpu().numpy().tolist()
                
                embeddings.append(query_embedding)
        
        if 'dpr' in model_code:
             # 保存时保持 list 结构以匹配原逻辑
             pass 
        else:
             embeddings = torch.stack(embeddings, dim=0).to(device)

    elif 'realm' in model_code or 'orqa' in model_code:
        for sample in tqdm(database_samples):
            ego = sample["ego"]
            perception = sample["perception"]
            prompt = f"{ego} {perception}"
            tokenized_input = tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            with torch.no_grad():
                input_ids = tokenized_input["input_ids"].to(device)
                attention_mask = tokenized_input["attention_mask"].to(device)
                query_embedding = model(input_ids, attention_mask).pooler_output
                embeddings.append(query_embedding)
        
        embeddings = torch.stack(embeddings, dim=0).to(device)

    # 保存缓存
    with open(cache_filename, "wb") as f:
        pickle.dump(embeddings, f)
    
    # 统一返回格式
    if isinstance(embeddings, list):
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    db_embeddings = embeddings.squeeze(1)

    return db_embeddings


def bert_get_adv_emb(data, model, tokenizer, num_adv_passage_tokens, adv_passage_ids, adv_passage_attention, device=device):
    """
    Optimized version of bert_get_adv_emb with batch processing for both QA and AD tasks.
    """
    # ------------------------------------------------------------------
    # 1. Data Preprocessing (Handle different Agent types)
    # ------------------------------------------------------------------
    
    # CASE A: QA Task (StrategyQA)
    if "question" in data.keys():
        text_input = data["question"]
    
    # CASE B: AD Task (AgentDriver) ---> 这是你需要添加的部分
    elif "ego" in data.keys() and "perception" in data.keys():
        # AgentDriver 的输入分为 ego（自身状态）和 perception（感知信息）
        # 原理：utils.py 中是 f"{ego} {perception}"
        # 优化：使用列表推导式一次性处理整个 batch，而不是写慢速循环
        text_input = [f"{ego} {perception}" for ego, perception in zip(data["ego"], data["perception"])]
        
    else:
        raise ValueError(f"Unrecognized data keys: {data.keys()}")

    # ------------------------------------------------------------------
    # 2. Parallel Tokenization (Batch Level)
    # ------------------------------------------------------------------
    # 一次性 Tokenize 整个 Batch，而不是一个个做
    
    with torch.no_grad():
        tokenized_input = tokenizer(
            text_input, 
            padding='max_length', 
            truncation=True, 
            max_length=512 - num_adv_passage_tokens, # 留出位置给 trigger
            return_tensors="pt"
        )
        
        input_ids = tokenized_input["input_ids"].to(device)
        attention_mask = tokenized_input["attention_mask"].to(device)
        # print(input_ids.shape) # batch_size * x
        # print(attention_mask.shape) # batch_size * x
        # print(adv_passage_ids.shape) # cand_size * token_size
        # print(adv_passage_attention.shape) # numcand * token
        
        # ------------------------------------------------------------------
        # 3. Trigger Insertion (Tensor Broadcasting)
        # ------------------------------------------------------------------
        # 获取当前 Batch 大小
        batch_size = input_ids.shape[0]
        candidate_size = adv_passage_ids.shape[0]

        if candidate_size != 1:
            adv_passage_ids = adv_passage_ids.unsqueeze(0) # 1*numcand * token
            adv_passage_attention = adv_passage_attention.unsqueeze(0) # 1 * numcand * token

            input_ids = input_ids.unsqueeze(1) # batch * 1 * x
            attention_mask = attention_mask.unsqueeze(1) # batch * 1 * x

            # 核心优化：利用 repeat 将 trigger 扩展到和 batch 一样大
            # 避免了在循环中手动拼接
            current_adv_ids = adv_passage_ids.repeat(batch_size,1, 1)          # [batch, num_cand, trigger_len]
            current_adv_attn = adv_passage_attention.repeat(batch_size, 1, 1)   # [batch, num_cand, trigger_len]
            input_ids = input_ids.repeat(1, candidate_size, 1) # batch, numcand, x
            attention_mask = attention_mask.repeat(1, candidate_size, 1) # batch, numcand, x

            current_adv_ids = current_adv_ids.view(-1, current_adv_ids.shape[-1])
            current_adv_attn = current_adv_attn.view(-1, current_adv_attn.shape[-1])
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            
            # 将 Trigger 拼接到原始输入的末尾 (Suffix Trigger)
            # [batch, seq_len] + [batch, trigger_len] -> [batch, seq_len + trigger_len]
            suffix_adv_passage_ids = torch.cat((input_ids, current_adv_ids), dim=1)
            suffix_adv_passage_attention = torch.cat((attention_mask, current_adv_attn), dim=1)
            # print(suffix_adv_passage_ids.shape) # batch * numcand * 512, as specified in tokenizer
        
        else:
            # 核心优化：利用 repeat 将 trigger 扩展到和 batch 一样大
            # 避免了在循环中手动拼接
            current_adv_ids = adv_passage_ids.repeat(batch_size, 1)          # [batch, trigger_len]
            current_adv_attn = adv_passage_attention.repeat(batch_size, 1)   # [batch, trigger_len]
            
            # 将 Trigger 拼接到原始输入的末尾 (Suffix Trigger)
            # [batch, seq_len] + [batch, trigger_len] -> [batch, seq_len + trigger_len]
            suffix_adv_passage_ids = torch.cat((input_ids, current_adv_ids), dim=1)
            suffix_adv_passage_attention = torch.cat((attention_mask, current_adv_attn), dim=1)

        
        # 构造模型输入字典
        p_sent = {
            'input_ids': suffix_adv_passage_ids, 
            'attention_mask': suffix_adv_passage_attention
        }
        
    # ------------------------------------------------------------------
    # 4. Forward Pass
    # ------------------------------------------------------------------
    # 兼容 DataParallel 和普通模型
    if isinstance(model, torch.nn.DataParallel):
            p_emb = model.module.bert(**p_sent).pooler_output
    elif hasattr(model, "bert"):
            p_emb = model.bert(**p_sent).pooler_output
    else:
            p_emb = model(**p_sent).pooler_output
            
    return p_emb # (batch*numcand) * token