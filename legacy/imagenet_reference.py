import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# TimM (PyTorch Image Models) for Modern Augmentations (Mixup, CutMix)
from timm.data import create_transform
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.data.mixup import Mixup

# --- CONFIGURATION (Matches LaViT/DeiT Paper Recipe) ---
CONFIG = {
    "epochs": 300,
    "batch_size": 256,         # Per GPU (Total effective = 1024 on 4 GPUs)
    "base_lr": 5e-4,           # 0.0005 for batch 512/1024 (Scale this if only using 1 GPU!)
    "weight_decay": 0.05,      # Standard AdamW weight decay
    "input_size": 224,
    "num_classes": 1000,
    "mixup_alpha": 0.8,        # Mixup strength
    "cutmix_alpha": 1.0,       # CutMix strength
    "mixup_prob": 1.0,         # Probability of applying mixup/cutmix
    "label_smoothing": 0.1,
    "num_workers": 8,
    "print_freq": 100
}

def setup_ddp():
    """Initializes the distributed backend (works for 1 GPU or Multi-GPU)."""
    # These env vars are set automatically by 'torchrun'
    if "RANK" in os.environ:
        init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Fallback for non-torchrun execution (debugging)
        print("Not using DDP. Running on single GPU.")
        return 0, 0, 1

def cleanup_ddp():
    if "RANK" in os.environ:
        destroy_process_group()

def build_loaders(data_dir, batch_size, input_size, world_size, rank):
    """
    Creates DataLoaders with Modern Augmentations (RandAugment).
    """
    # 1. Train Transforms (DeiT / TimM Style: RandAugment + AutoAugment)
    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1', # Standard DeiT RandAugment policy
        interpolation='bicubic',
        re_prob=0.25,                        # Random Erasing
        re_mode='pixel',
        re_count=1,
    )

    # 2. Validation Transforms (Standard Resize + CenterCrop)
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Datasets
    # Assumes structure: /path/to/imagenet/train/class_folders/images
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)

    # 4. Samplers (Crucial for DDP)
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # 5. Loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=CONFIG["num_workers"], pin_memory=True, sampler=train_sampler, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=int(batch_size * 1.5), shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True, sampler=val_sampler
    )

    return train_loader, val_loader, train_sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet data')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()

    # 1. DDP Setup
    rank, local_rank, world_size = setup_ddp()
    
    # Scale LR based on total batch size (Linear Scaling Rule)
    # Base LR 5e-4 is for batch 512. If batch is different, scale.
    total_batch_size = CONFIG["batch_size"] * world_size
    actual_lr = CONFIG["base_lr"] * (total_batch_size / 512.0)

    if rank == 0:
        print(f"Training ResNet-50 on ImageNet")
        print(f"World Size: {world_size} | Total Batch: {total_batch_size} | LR: {actual_lr:.6f}")

    # 2. Model Setup
    model = models.resnet50(weights=None) # From scratch
    model = model.cuda(local_rank)

    # Convert BatchNorm to SyncBatchNorm (Critical for DDP accuracy)
    if world_size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    # 3. Optimizer (AdamW as per LaViT/DeiT paper)
    optimizer = optim.AdamW(model.parameters(), lr=actual_lr, weight_decay=CONFIG["weight_decay"])

    # 4. Mixup / CutMix Function
    mixup_fn = Mixup(
        mixup_alpha=CONFIG["mixup_alpha"], 
        cutmix_alpha=CONFIG["cutmix_alpha"], 
        prob=CONFIG["mixup_prob"], 
        switch_prob=0.5, 
        mode='batch',
        label_smoothing=CONFIG["label_smoothing"], 
        num_classes=CONFIG["num_classes"]
    )

    # 5. Loss Function
    # SoftTargetCrossEntropy is required because Mixup labels are not integers anymore
    train_criterion = SoftTargetCrossEntropy().cuda(local_rank)
    val_criterion = nn.CrossEntropyLoss().cuda(local_rank)

    # 6. Data Loaders
    train_loader, val_loader, train_sampler = build_loaders(
        args.data_dir, CONFIG["batch_size"], CONFIG["input_size"], world_size, rank
    )

    # 7. LR Scheduler (Cosine with Warmup)
    scheduler = CosineLRScheduler(
        optimizer, t_initial=CONFIG["epochs"], warmup_t=5, warmup_lr_init=1e-6, cycle_limit=1
    )

    # --- Training Loop ---
    start_epoch = 0
    
    # Resume logic
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0: print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, CONFIG["epochs"]):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        model.train()
        optimizer.zero_grad()
        
        num_steps = len(train_loader)
        scheduler.step_update(epoch * num_steps) # Initialize scheduler for the epoch

        for i, (images, target) in enumerate(train_loader):
            images, target = images.cuda(local_rank, non_blocking=True), target.cuda(local_rank, non_blocking=True)

            # Apply Mixup/CutMix
            images, target = mixup_fn(images, target)

            outputs = model(images)
            loss = train_criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step scheduler every iteration (Cosine Decay is continuous)
            scheduler.step_update(epoch * num_steps + i)

            if i % CONFIG["print_freq"] == 0 and rank == 0:
                print(f"Epoch [{epoch}/{CONFIG['epochs']}] Step [{i}/{num_steps}] Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- Validation (Only runs on original targets, no mixup) ---
        if rank == 0: # Simple validation on rank 0 (approximate) or gather all (complex)
            validate(model, val_loader, val_criterion, epoch)
            
            # Save Checkpoint
            checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
            save_dict = {
                'epoch': epoch,
                'model': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(save_dict, checkpoint_path)

    cleanup_ddp()

def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    acc = 100. * correct / total
    print(f"--- VALIDATION Epoch {epoch} --- Acc: {acc:.2f}% | Loss: {total_loss/len(loader):.4f}")

if __name__ == '__main__':
    main()