import os
from huggingface_hub import snapshot_download

# 設定快取目錄與模型名稱
cache_dir = "./qwan_models_cach"
model_id = "Qwen/Qwen3-ASR-1.7B"

print(f"開始下載 {model_id} 到 {cache_dir}，這可能需要一些時間...")

# 下載模型
snapshot_download(
    repo_id=model_id,
    cache_dir=cache_dir,
    # 如果你在 Windows 上遇到軟連結問題，可以取消註解下一行
    # local_dir_use_symlinks=False 
)

print(f"模型已成功下載並快取至 {cache_dir}！")
