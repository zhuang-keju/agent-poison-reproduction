import gdown
import os
from pathlib import Path


def download_agent_driver(folder_id):

    if os.path.exists("./agentdriver/data/finetune"):
        print(f"✅ 检测到 './agentdriver/data/finetune' 文件夹已存在，跳过下载和解压。")
        return
        
    
    # 2. 下载
    url = f'https://drive.google.com/drive/folders/{folder_id}'
    output_folder = "./agentdriver"

    print(f"⬇️ Downloading AgentDriver dataset...")
    gdown.download_folder(url, output=output_folder, quiet=False, use_cookies=False)
    
    # 3. 直接解压到当前目录
    print("📦 Unzipping...")
    # -o: 覆盖不提示, -q: 安静模式
    !unzip -o -q {output_file}
    
    # 4. 简单确认
    if os.path.exists(output_folder):
        print(f"✅ 下载完成！文件夹已保存在: {output_folder}")
        # 检查一下内部结构
        print("📁 文件夹内容预览:", os.listdir(output_folder))
    else:
        print("❌ 下载似乎未成功，请检查 Folder ID 是否正确，或该文件夹是否设为 '任何拥有链接的人可见'。")

    
    # if os.path.exists("./agentdriver/data"):
    #     print("✅ 成功！数据集已就绪: ./agentdriver/data")
    # else:
    #     print("⚠️ 解压完成。请检查左侧文件栏，确认解压出的文件夹名是否为 agentdriver。")
    #     # 如果解压出来叫 data，你可能需要手动改名为 agentdriver/data
    #     print(f"当前目录下的文件夹: {[d for d in os.listdir() if os.path.isdir(d)]}")
    
    # 清理压缩包
    # if os.path.exists(output_file):
        # os.remove(output_file)

# ==========================================
# 👇 这里填入 AgentDriver 的 Google File ID
folder_id = '1ZrSZfTlH347hNoKADY3WjN0usmK9Dgt2' 
# ==========================================

%cd /kaggle/working/AgentPoison

download_agent_driver(folder_id)


def list_files_pathlib(directory):
    # rglob('*') 表示递归查找所有文件和文件夹
    # 如果只要文件，可以在后面加 if f.is_file()
    return [str(f) for f in Path(directory).rglob('*') if f.is_file()]

# 使用
files = list_files_pathlib("./agentdriver/data")
for f in files:
    if "DS_Store" not in f:
        print(f)