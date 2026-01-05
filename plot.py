# --- [新增] 1. 导入 PCA 和绘图库 ---
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import numpy as np



# --- [新增] 2. 定义 plot_PCA 函数 ---
# with random sampling, not entire db
def plot_PCA(query_embeddings, db_embeddings, root_dir, title):
    """
    对 query_embeddings (攻击样本) 和 db_embeddings (良性样本) 进行 PCA 降维并绘图
    """
    # 1. 转换为 CPU numpy 数组 (为了 sklearn)
    # 假设输入可能是 tensor，先 detach 转 CPU
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if isinstance(db_embeddings, torch.Tensor):
        db_embeddings = db_embeddings.detach().cpu().numpy()

    # 2. 初始化 PCA
    pca = PCA(n_components=2)
    
    # 3. 拼接数据进行统一降维 (保证坐标系一致)
    # 注意：db_embeddings 可能很大 (几万条)，为了速度和显示效果，
    # 我们可以只随机采样一部分 db_embeddings 画背景 (例如 2000 个点)
    if len(db_embeddings) > 2000:
        indices = np.random.choice(len(db_embeddings), 2000, replace=False)
        db_subset = db_embeddings[indices]
    else:
        db_subset = db_embeddings

    all_embeddings = np.vstack((query_embeddings, db_subset))
    
    # 4. 执行 PCA
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # 5. 拆分回两组
    reduced_query = reduced_embeddings[:len(query_embeddings)]
    reduced_db = reduced_embeddings[len(query_embeddings):]

    # 6. 绘图
    plt.figure(figsize=(10, 8))
    # 画背景 (良性样本 - 灰色)
    plt.scatter(reduced_db[:, 0], reduced_db[:, 1], c='grey', alpha=0.3, label='Benign Embeddings', s=10)
    # 画当前样本 (攻击样本 - 红色)
    plt.scatter(reduced_query[:, 0], reduced_query[:, 1], c='red', alpha=0.8, label='Adversarial Embeddings', s=20)
    
    plt.title(f'PCA of Embeddings {title}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. 保存
    save_path = f"{root_dir}/pca_{title.replace(' ', '_')}.png"
    plt.savefig(save_path)
    plt.show()
    plt.close() # 关闭画布释放内存
    # print(f"📊 Plot saved to {save_path}")


# --- [新增] A. 全量计算 Embeddings 的函数 ---
def get_full_dataset_embeddings(dataloader, model, tokenizer, num_adv_tokens, adv_ids, adv_attn, device):
    """
    遍历整个 DataLoader，计算所有样本在当前 Trigger 下的 Embeddings。
    """
    all_embeddings = []
    
    # 临时切换到评估模式，不计算梯度以节省显存
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # 使用 autocast 加速并减少显存占用
            with torch.cuda.amp.autocast():
                # 注意：这里调用的是 utils.py 或 notebook 里定义的 bert_get_adv_emb
                # 请确保该函数在当前上下文中可用
                emb = bert_get_adv_emb(
                    batch, 
                    model, 
                    tokenizer, 
                    num_adv_tokens, 
                    adv_ids, 
                    adv_attn, 
                    device=device
                )
            all_embeddings.append(emb.cpu()) # 及时转到 CPU 防止显存爆炸
            
    # 拼接所有 batch
    return torch.cat(all_embeddings, dim=0)


# coordinates only on db embedding, not query + db
# --- [修改版] 固定坐标系的 PCA 绘图函数 ---
def plot_PCA_full_only_db(query_embeddings, db_embeddings, root_dir, title):
    """
    1. 仅使用 db_embeddings (良性样本) 来拟合 PCA 坐标系 (Fit)。
    2. 将 query_embeddings (攻击样本) 投影 (Transform) 到该坐标系中。
    这样可以保证背景形状不变，清晰展示红点的移动过程。
    """
    # 1. 转换数据类型
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if isinstance(db_embeddings, torch.Tensor):
        db_embeddings = db_embeddings.detach().cpu().numpy()

    # 2. 初始化 PCA
    pca = PCA(n_components=2)
    
    # === 关键修改点 ===
    # 只用良性样本来定坐标系 (Fit)
    # 这样灰色点的形状会展开，不会因为红点跑远了而被压扁
    pca.fit(db_embeddings)
    
    # 3. 分别进行投影 (Transform)
    reduced_db = pca.transform(db_embeddings)
    reduced_query = pca.transform(query_embeddings)
    # =================
    
    # 4. 绘图
    plt.figure(figsize=(12, 10)) 
    
    # 画背景 (良性样本 - 灰色)
    # 稍微调大一点 alpha 让云团结构更明显
    plt.scatter(reduced_db[:, 0], reduced_db[:, 1], c='grey', alpha=0.3, label='Benign Database', s=15)
    
    # 画当前样本 (攻击样本 - 红色)
    plt.scatter(reduced_query[:, 0], reduced_query[:, 1], c='red', alpha=0.6, label='Adversarial Queries', s=20)
    
    plt.title(f'PCA - {title}\n(Fixed Basis on Benign DB)')
    plt.xlabel('PC 1 (Benign Variance)')
    plt.ylabel('PC 2 (Benign Variance)')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 5. 保存与显示
    save_path = f"{root_dir}/pca_fixed_{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150)
    plt.show() 
    plt.close()




# this function does PCA on both the query and db embedding together
# the result is that red and grey are separated apart naturally

# --- [修改] B. 无采样的 PCA 绘图函数 ---
def plot_PCA_full(query_embeddings, db_embeddings, root_dir, title):
    """
    对全量数据进行 PCA 并绘图。
    """
    # 1. 转换数据类型
    if isinstance(query_embeddings, torch.Tensor):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if isinstance(db_embeddings, torch.Tensor):
        db_embeddings = db_embeddings.detach().cpu().numpy()

    # 2. 数据准备：不再采样，直接使用全部数据
    # 拼接 (前景 + 背景)
    all_embeddings = np.vstack((query_embeddings, db_embeddings))
    
    # 3. PCA 降维
    # 如果数据量非常大 (>10w)，这一步可能会慢，StrategyQA (几千条) 没问题
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # 4. 拆分
    reduced_query = reduced_embeddings[:len(query_embeddings)]
    reduced_db = reduced_embeddings[len(query_embeddings):]

    # 5. 绘图
    plt.figure(figsize=(12, 10)) # 画布调大一点
    
    # 画背景 (良性样本 - 灰色)
    # alpha 调低一点 (0.1)，因为点多了会重叠，透明度低能看出密度
    plt.scatter(reduced_db[:, 0], reduced_db[:, 1], c='grey', alpha=0.15, label='Benign Database', s=10)
    
    # 画当前样本 (攻击样本 - 红色)
    plt.scatter(reduced_query[:, 0], reduced_query[:, 1], c='red', alpha=0.4, label='Adversarial Queries', s=15)
    
    plt.title(f'PCA - {title}\n(Red: {len(reduced_query)} queries, Grey: {len(reduced_db)} db samples)')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # 6. 保存与显示
    save_path = f"{root_dir}/pca_full_{title.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=150) # dpi 调高一点更清晰
    plt.show() # 在 Notebook 中显示
    plt.close()



def example_plot(agent_name, db_embeddings, model_name):
    # 1. 确保 args.agent 设置正确
    # args.agent = "ad"
    print(f"🎨 Testing Background Visualization for Agent: {agent_name}")
    
    # 2. 加载数据库 Embeddings (良性样本)
    # 这步会用到你刚刚修改好的路径逻辑
    # if args.agent == "ad":
    #     database_samples_dir = "agentdriver/data/finetune/data_samples_train.json"
    #     db_dir = "agentdriver/data/memory"
    #     # 加载 (如果之前内存不够，这里可能会有点慢，大概几秒到几十秒)
    #     print("⏳ Loading DB embeddings...")
    #     db_embeddings = load_db_ad_simple(database_samples_dir, db_dir, args.model, model, tokenizer, device)
        
    #     # 如果 load_db_ad 返回的是 tuple，取第一个；如果是 tensor 直接用
    #     if isinstance(db_embeddings, tuple):
    #         db_embeddings = db_embeddings[0]
    print(f"✅ DB Embeddings Loaded. Shape: {db_embeddings.shape}")
    
    # 3. 执行 PCA (只对背景)
    print("🧮 Running PCA on benign samples...")
    db_np = db_embeddings.detach().cpu().numpy()
    
    # 为了画图好看，如果数据量太大 (>10000)，可以随机采样一部分，或者全画
    # AgentDriver 全量大概有几万条，全画出来效果最好（只要不卡死浏览器）
    if len(db_np) > 20000:
        print(f"  (Sampling 20000 points from {len(db_np)} for visualization speed)")
        indices = np.random.choice(len(db_np), 20000, replace=False)
        db_viz = db_np[indices]
    else:
        db_viz = db_np
    
    pca = PCA(n_components=2)
    reduced_db = pca.fit_transform(db_viz)
    
    # 4. 绘图 (纯灰色模式)
    plt.figure(figsize=(12, 10))
    plt.scatter(reduced_db[:, 0], reduced_db[:, 1], c='grey', alpha=0.3, s=10, label='Benign Samples (AgentDriver)')
    
    plt.title(f'AgentDriver Background Distribution\n(Model: {model_name})')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig('island_shape.png', dpi=300, bbox_inches='tight')
    plt.show()
