import os
import torch
import timm
from urllib.request import urlretrieve

def download_convnext_weights():
    # 저장할 폴더 생성
    save_dir = "./model/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "convnextv2_base.pt")

    print("Checking model url...")
    model_name = 'convnextv2_base.fcmae_ft_in22k_in1k'
    cfg = timm.models.get_pretrained_cfg(model_name)
    url = cfg.url
    
    print(f"Downloading weights from: {url}")
    print(f"Saving to: {save_path}")

    # 다운로드 진행
    try:
        urlretrieve(url, save_path)
        print("Download Complete!")
    except Exception as e:
        print(f"Download failed: {e}")
        print("수동 다운로드가 필요할 수 있습니다.")

if __name__ == "__main__":
    download_convnext_weights()