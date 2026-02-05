'''Training & Evaluation Module for Convolutional Neural Networks'''

# Utility Imports 
import os 
import time 

# Torch 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.profiler 
import torch.distributed as dist # Data Parallel

# Timm + Huggingface for Swin Transformer Training 
from timm.loss import SoftTargetCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from transformers import get_linear_schedule_with_warmup    

# Custom Imports
from utils import set_seed


""" Setup distributed training environment """
def setup_distributed():
    # Torchrun sets the following environment variables
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        word_size = int(os.environ["WORLD_SIZE"])

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        return local_rank 
    else: 
        print("Not using Distributed Data Parallel Mode")
        return 0 

""" Cleanup distributed training environment """
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

""" Reducing tensor across all processes """
def reduce_tensor(tensor):
    rt = tensor.clone() 
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt 
  
def Train_Eval(args, 
               model: nn.Module, 
               train_loader, 
               test_loader
               ):
    
    if args.seed != 0:
        set_seed(args.seed)

    # Loss Criterion
    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()
            
    # Optimizer 
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        

    # Learning Rate Scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        
    # Device
    device = args.device
    model.to(device)
    criterion.to(device)

    scaler = GradScaler() if args.use_amp else None 
        
    epoch_results = []

    ## [GFLOPS] Computation using PyTorch Profiler ##
    try:
        model.eval()
        input_tensor, _ = next(iter(train_loader))
        input_tensor = input_tensor.to(device)

        # Profile a single forward pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                model(input_tensor[0:1])

        total_flops = sum(event.flops for event in prof.key_averages())
        if total_flops > 0:
            gflops = total_flops / 1e9
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            print(f"Model Complexity (Profiler):")
            print(f"   - Total Parameters: {params:.8f} M")
            print(f"   - GFLOPs: {gflops:.8f}")
            epoch_results.append(f"Model Complexity (Profiler): GFLOPs: {gflops:.8f}, Trainable Parameters: {params:.8f} M")
    except Exception as e:
        print(f"Could not calculate GFLOPs with PyTorch Profiler: {e}")
    
    # Compile Model 
    if args.compile: 
        model = torch.compile(
            model, 
            mode=args.compile_mode, 
            fullgraph=False, 
            dynamic=False) 
        print("compiled success!")
        
    # Training Loop
    epoch_times = [] # Average Epoch Time 
    max_accuracy = 0.0 
    max_epoch = 0
    
    for epoch in range(args.num_epochs):
        # Model Training
        model.train() 
        train_running_loss = 0.0
        test_running_loss = 0.0
        
        start_time = time.time()

        train_top1_5 = [0, 0]
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # use mixed precision training
            if args.use_amp:
                with autocast(device_type=args.device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if args.clip_grad_norm:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:    
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if args.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()            

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            train_top1_5[0] += top1.item()
            train_top1_5[1] += top5.item()
            train_running_loss += loss.item()

        train_top1_5[0] /= len(train_loader)
        train_top1_5[1] /= len(train_loader)
        train_running_loss /= len(train_loader)
        
        # Model Evaluation 
        model.eval()
        test_top1_5 = [0, 0]
        with torch.no_grad():
            for images, labels in test_loader: 
                images, labels = images.to(device), labels.to(device)
                if args.use_amp:
                    with autocast(device_type=args.device):
                        outputs = model(images)
                else: 
                    outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                test_top1_5[0] += top1.item()
                test_top1_5[1] += top5.item()
        
        test_top1_5[0] /= len(test_loader)
        test_top1_5[1] /= len(test_loader)
        test_running_loss /= len(test_loader)

        # Single Epoch Duration
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Save Epoch Results
        epoch_results.append(f"[Epoch {epoch+1:03d}] Time: {epoch_time:.4f}s | [Train] Loss: {train_running_loss:.8f} Accuracy: Top1: {train_top1_5[0]:.4f}%, Top5: {train_top1_5[1]:.4f}% | [Test] Loss: {test_running_loss:.8f} Accuracy: Top1: {test_top1_5[0]:.4f}%, Top5: {test_top1_5[1]:.4f}%")
        print(epoch_results[-1])
        
        # Max Accuracy Check
        if test_top1_5[0] > max_accuracy:
            max_accuracy = test_top1_5[0]
            max_epoch = epoch + 1    
            
        # Learning Rate Scheduler Step
        if scheduler: 
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_top1_5[0])
            else:
                scheduler.step()
                
                
    epoch_results.append(f"\nAverage Epoch Time: {sum(epoch_times) / len(epoch_times):.4f}s")
    epoch_results.append(f"Max Accuracy: {max_accuracy:.4f}% at Epoch {max_epoch}")
    
    return epoch_results

"""
Training & Evaluation Loop for ImageNet1K Classification
- Same training setting as Swin Transformer (Liu et al., 2021b). 
- AdamW Optimizer + Cosine LR Scheduler + SoftTargetCrossEntropy Loss
- Lr = 1e-3, Weight Decay = 0.05, Epochs = 300, Batch Size = 1024
- Image Size = 224x224, Mixup & CutMix Augmentations

Swin Transformer Config for Hyperparameters
https://github.com/microsoft/Swin-Transformer/blob/main/config.py
"""
def Train_Eval_ImageNet(args, 
                        model: nn.Module, 
                        train_loader, 
                        test_loader, 
                        train_sampler,
                        mixup_fn=None, 
                        rank=0
                        ):

    # Set Seed
    if args.seed != 0:
        set_seed(args.seed)

    # Loss Criterion
    train_criterion = SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss()  # Using Timm's SoftTargetCrossEntropy for ImageNet
    eval_criterion = nn.CrossEntropyLoss()

    # Optimizer 
    optimizer = optim.AdamW(
        params=model.parameters(), 
        lr=1e-3, 
        weight_decay=0.05, 
        eps=1e-8, 
        betas=(0.9, 0.999)
        )
    
    # Scheduler 
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=300,
        lr_min=5e-6,
        warmup_lr_init=5e-7,
        warmup_t=20,
        cycle_limit=1,
        t_in_epochs=True,
    )

    args.num_epochs = 300
    args.clip_grad_norm = 5.0
    args.use_amp = True 
    

    # Device 
    device = args.device 
    model.to(device) 
    train_criterion.to(device)
    eval_criterion.to(device) 

    # H100 Optimization 
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler() if (args.use_amp and not use_bf16) else None
    
    epoch_results = [] 

    ## [GFLOPS] Computation using PyTorch Profiler ##
    try:
        model.eval()
        batch = next(iter(train_loader))
        input_tensor = batch['pixel_values'].to(device)

        # Profile a single forward pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                model(input_tensor[0:1])

        total_flops = sum(event.flops for event in prof.key_averages())
        
        if total_flops > 0:
            gflops = total_flops / 1e9
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            print(f"Model Complexity (Profiler):")
            print(f"   - Total Parameters: {params:.8f} M")
            print(f"   - GFLOPs: {gflops:.8f}")
            epoch_results.append(f"Model Complexity (Profiler): GFLOPs: {gflops:.8f}, Trainable Parameters: {params:.8f} M")
    except Exception as e:
        print(f"Could not calculate GFLOPs with PyTorch Profiler: {e}")

    # Compile Model 
    if args.compile: 
        model = torch.compile( 
            model=model, 
            mode=args.compile_mode, 
            fullgraph=False, 
            dynamic=False)
        print("compiled success!")
        
    # Training Loop 
    epoch_times = [] # Average Epoch Time 
    max_accuracy = 0.0 
    max_epoch = 0 

    for epoch in range(args.num_epochs):
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)
        
        # Model Training 
        model.train() 
        train_running_loss = 0.0
        test_running_loss = 0.0
        start_time = time.time() 
        train_top1_5 = [0, 0]
        
        for i, batch in enumerate(train_loader): 
            if isinstance(batch, dict):
                images = batch['pixel_values'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
            else: 
                images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                
            # Apply Mixup if available
            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            optimizer.zero_grad() 
            
            # use mixed precision training
            if args.use_amp:
                with autocast(device_type=args.device, dtype=dtype):
                    outputs = model(images)                    
                    loss = train_criterion(outputs, labels)

                if scaler is not None: # FP16 Path 
                    scaler.scale(loss).backward()
                    if args.clip_grad_norm:
                        scaler.unscale_(optimizer) # Unscale gradients before clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else: # BF16 Path
                    loss.backward()
                    if args.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                    optimizer.step()
            else:    
                outputs = model(images)
                loss = train_criterion(outputs, labels)
                loss.backward()
                if args.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()            

            # Accuracy calculation with mixup labels is approximate 
            if mixup_fn is None: 
                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                train_top1_5[0] += top1.item()
                train_top1_5[1] += top5.item()
            train_running_loss += loss.item()


        if mixup_fn is None: 
            train_top1_5[0] /= len(train_loader)
            train_top1_5[1] /= len(train_loader)
        train_running_loss /= len(train_loader)

        ## Validation Phase 
        test_top1_5 = [0, 0]
        test_running_loss = 0.0

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if isinstance(batch, dict):
                    images = batch['pixel_values'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)
                else: 
                    images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

                if args.use_amp:
                    with autocast(device_type=args.device, dtype=dtype):
                        outputs = model(images)
                else: 
                    outputs = model(images)
                
                loss = eval_criterion(outputs, labels)
                test_running_loss += loss.item()

                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                test_top1_5[0] += top1.item()
                test_top1_5[1] += top5.item()

        test_top1_5[0] /= len(test_loader)
        test_top1_5[1] /= len(test_loader)
        test_running_loss /= len(test_loader)

        # Aggregate metrics across all processes
        val_metrics = torch.tensor([test_top1_5[0], test_top1_5[1], test_running_loss], device=device)
        if dist.is_initialized():
            dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM) # Sum across all processes
            val_metrics /= dist.get_world_size() # Average

        test_top1_5[0] = val_metrics[0].item()
        test_top1_5[1] = val_metrics[1].item()
        test_running_loss = val_metrics[2].item()

        # Single Epoch Duration
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Save Epoch Results
        result_str = f"[Epoch {epoch+1:03d}] Time: {epoch_time:.4f}s | [Train] Loss: {train_running_loss:.8f} Accuracy: Top1: {train_top1_5[0]:.4f}%, Top5: {train_top1_5[1]:.4f}% | [Test] Loss: {test_running_loss:.8f} Accuracy: Top1: {test_top1_5[0]:.4f}%, Top5: {test_top1_5[1]:.4f}%"
        epoch_results.append(result_str)

        if test_top1_5[0] > max_accuracy:
            max_accuracy = test_top1_5[0]
            max_epoch = epoch + 1
            
        # Print only for rank 0
        if rank == 0:
            print(result_str)

        # Learning Rate Scheduler Step
        scheduler.step(epoch + 1)

    if rank == 0: 
        epoch_results.append(f"\nAverage Epoch Time: {sum(epoch_times) / len(epoch_times):.4f}s")
        epoch_results.append(f"Max Accuracy: {max_accuracy:.4f}% at Epoch {max_epoch}")

        # Saving model 
        save_path = os.path.join(args.output_dir, f"final_model.pth")
        model_to_save = model.module if hasattr(model, 'module') else model

        save_model(model_to_save, args, optimizer, scheduler, epoch, test_top1_5[0], max_accuracy, save_path)

    return epoch_results

def accuracy(output, target, topk=(1,)):
    """Computes the top-1 and top-5 accuracy of the model."""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk] # [72.5, 91.3] - [top1, top5]

""" Model Save & Load Functions """
def save_model(model, args, optimizer, scheduler, epoch, last_accuracy, best_accuracy, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'last_accuracy': last_accuracy,
        'best_accuracy': best_accuracy,
        'args': args
    }, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

def load_model(model, checkpoint_path, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")

    print(f"Loaded model from epoch {checkpoint['epoch']} with best accuracy {checkpoint['best_accuracy']:.4f}%")
    return model