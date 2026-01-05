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