import os

os.environ["DECORD_LOG_LEVEL"] = "FATAL"
import sys
import re
import yaml
import argparse
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from torchvision import transforms
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from src.models import DeepfakeClassifierDINOv3, DeepfakeClassifierConvNeXtV2
from tqdm.auto import tqdm

from src.dataset import Celeb_DF, FaceForensics, DFDC, WildDeepfake
from src.utils import RandomJPEGCompression, get_llrd_params_dinov3, get_llrd_params_convnext, split_faceforensics, split_celeb_df, split_dfdc, split_wilddeepfake

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



def train_one_epoch(epoch, total_epochs, model, dataloader, optimizer, criterion, accelerator):
    model.train()
    epoch_loss = 0.0

    progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Epoch {epoch}/{total_epochs}", file=sys.stdout)
        
    for batch_idx, (images, labels) in enumerate(progress_bar):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = epoch_loss / len(dataloader)
    accelerator.print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, dataloader, criterion, accelerator):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]
            probs_gathered, labels_gathered = accelerator.gather_for_metrics((probs, labels))
            
            all_probs.extend(probs_gathered.cpu().tolist())
            all_labels.extend(labels_gathered.cpu().tolist())
            
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, all_labels, all_probs

def main():
    parser = argparse.ArgumentParser(description="Deepfake DINOv3 Training Script")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="all", 
                            project_dir="logs", 
                            kwargs_handlers=[ddp_kwargs], 
                            mixed_precision="bf16"
                            )

    set_seed(42)
    if accelerator.is_main_process:
        accelerator.print(f"Accelerator Initialized. Device: {accelerator.device}, Num Processes: {accelerator.num_processes}")

    # Config
    train_cfg = config['train']
    data_cfg = config['data']
    model_cfg = config['model']

    lr = float(train_cfg['lr'])
    batch_size = train_cfg['batch_size']
    epochs = train_cfg['epochs']
    img_size = data_cfg['image_size']
    
    train_data_path = data_cfg['train_data']
    eval_data_path = data_cfg['eval_data']
    frames_per_video = data_cfg.get('frames_per_video', 10)

    # Hyper Parameters
    checkpoint_path = model_cfg.get('checkpoint_path', '')

    accelerator.print(f"Loading Configuration:\n{config}")

    # Data Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.3),
        RandomJPEGCompression(quality_range=(30, 80), p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if accelerator.is_main_process:
        accelerator.print(f"Loading data from: {train_data_path}")

    # Datasets
    # =========================================================
    # Celeb-DF Datasets
    crop_root = os.path.join(train_data_path, "Dataset", "celeb_df")
    crop_train, crop_eval = split_celeb_df(train_data_path, seed=42)
    crop_train_dataset = Celeb_DF(crop_train, root_dir=crop_root, transform=train_transform)
    crop_eval_dataset = Celeb_DF(crop_eval, root_dir=crop_root, transform=eval_transform)

    # FaceForensics Split
    face_train, face_eval = split_faceforensics(train_data_path, seed=42)
    face_train_dataset = FaceForensics(root_dir=train_data_path, video_ids=face_train, transform=train_transform)
    face_eval_dataset = FaceForensics(root_dir=train_data_path, video_ids=face_eval, transform=eval_transform)

    # DFDC Split
    dfdc_train, dfdc_eval = split_dfdc(train_data_path, seed=42)
    dfdc_train_dataset = DFDC(root_dir=train_data_path, video_ids=dfdc_train, transform=train_transform)
    dfdc_eval_dataset = DFDC(root_dir=train_data_path, video_ids=dfdc_eval, transform=eval_transform)

    # WildDeepfake Split
    wild_train, wild_eval = split_wilddeepfake(train_data_path, seed=42)
    wild_train_dataset = WildDeepfake(root_dir=train_data_path, video_ids=wild_train, transform=train_transform)
    wild_eval_dataset = WildDeepfake(root_dir=train_data_path, video_ids=wild_eval, transform=eval_transform)
    # =========================================================

    train_datasets = [
        crop_train_dataset,
        face_train_dataset,
        dfdc_train_dataset,
        wild_train_dataset
    ]

    eval_datasets = [
        crop_eval_dataset,
        face_eval_dataset,
        dfdc_eval_dataset,
        wild_eval_dataset
    ]

    full_train_dataset = ConcatDataset(train_datasets)
    full_eval_dataset = ConcatDataset(eval_datasets)

    accelerator.print(f"Total Train Samples: {len(full_train_dataset)}")
    accelerator.print(f"Total Eval Samples: {len(full_eval_dataset)}")

    accelerator.print("Calculating global class weights...")

    total_real = 0
    total_fake = 0

    for ds in train_datasets:
        labels = [s['label'] for s in ds.samples]
        counts = Counter(labels)
        total_real += counts.get(0, 0)
        total_fake += counts.get(1, 0)

    accelerator.print(f"Global Counts -> Real(0): {total_real}, Fake(1): {total_fake}")

    loss_weights = None

    if total_real > 0 and total_fake > 0:
        total_count = total_real + total_fake
        
        w0 = total_count / (2 * total_real) # Real 가중치
        w1 = total_count / (2 * total_fake) # Fake 가중치
        
        loss_weights = torch.tensor([w0, w1], dtype=torch.float32).to(accelerator.device)
        
        accelerator.print(f"Calculated Loss Weights: {loss_weights}")
    else:
        accelerator.print("[Warning] 데이터가 없거나 한쪽 클래스가 0개입니다. 가중치를 적용하지 않습니다.")

    # DataLoader
    crop_train_loader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 8), 
        pin_memory=True
    )
    crop_eval_loader = DataLoader(
        full_eval_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=train_cfg.get('num_workers', 8), 
        pin_memory=True
    )

    # Model
    # model = DeepfakeClassifierDINOv3(
    #     model_name=model_cfg['name'], 
    #     num_classes=model_cfg['num_classes'], 
    #     checkpoint_path=model_cfg.get('checkpoint_path', ''),
    #     pretrained=False
    # )
    model = DeepfakeClassifierConvNeXtV2(
        model_name=model_cfg['name'], 
        num_classes=model_cfg['num_classes'], 
        checkpoint_path=model_cfg.get('checkpoint_path', ''),
        pretrained=True
    )
    
    # Optimizer
    for name, param in model.named_parameters():
        param.requires_grad = True
    # params = get_llrd_params(model, lr)
    params = get_llrd_params_convnext(model, lr)
    optimizer = optim.AdamW(params)

    # Scheduler & Loss
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Prepare for Distributed Training
    model, optimizer, crop_train_loader, crop_eval_loader, scheduler = accelerator.prepare(
        model, optimizer, crop_train_loader, crop_eval_loader, scheduler
    )

    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False

    accelerator.print("Starting training...")

    best_auc = 0.0
    for epoch in range(1, epochs+1):            
        if epoch == 11:
            accelerator.print("Unfreezing all parameters...")
            for name, param in model.named_parameters():
                param.requires_grad = True

        train_loss = train_one_epoch(epoch, epochs, model, crop_train_loader, optimizer, criterion, accelerator)
        scheduler.step()
        
        val_loss, val_labels, val_probs = validate(model, crop_eval_loader, criterion, accelerator)
        
        if accelerator.is_main_process:
            try:
                roc_auc = roc_auc_score(val_labels, val_probs)
                
                preds = [1 if p > 0.5 else 0 for p in val_probs]
                correct = sum([1 for p, l in zip(preds, val_labels) if p == l])
                acc = correct / len(val_labels)
                
                accelerator.print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f} | AUC: {roc_auc:.4f} | Acc: {acc:.4f}")

                if roc_auc > best_auc:
                    best_auc = roc_auc
                    output_dir = os.path.join("checkpoints", "best_model")
                    accelerator.save_state(output_dir)
                    accelerator.print(f"New Best AUC! Saved to {output_dir}")

                save_interval = train_cfg.get('log_interval', 10)
                if (epoch + 1) % save_interval == 0:
                    output_dir = os.path.join("checkpoints", f"epoch_{epoch+1}")
                    accelerator.save_state(output_dir)
            
            except Exception as e:
                accelerator.print(f"Metric calculation failed: {e}")

    accelerator.print("Training finished.")

if __name__ == "__main__":
    main()
