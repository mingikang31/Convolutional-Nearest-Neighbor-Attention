'''Training & Evaluation Module for Convolutional Neural Networks'''

import torch
import torch.nn as nn
import torch.optim as optim

import time 

from utils import set_seed



def Train_Eval(args, 
               model: nn.Module, 
               train_loader, 
               test_loader
               ):
    
    if args.seed != 0:
        set_seed(args.seed)
    
    if args.criterion == 'CrossEntropy':
        if args.layer == "ConvNNAttention" or args.layer == "ConvNN":
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            criterion = nn.CrossEntropyLoss()
        else: 
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
    
    if args.use_amp:
        scaler = torch.amp.GradScaler("cuda")
        
    epoch_results = []

    # ==================== ROBUST GFLOPs Calculation with PyTorch Profiler ====================
    try:
        import torch.profiler

        # Get a single batch from the train_loader to determine input size
        input_tensor, _ = next(iter(train_loader))
        input_tensor = input_tensor.to(device)

        # Profile a single forward pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                model(input_tensor[0:1])

        # A more robust way to get total FLOPs: sum them up from all events
        total_flops = sum(event.flops for event in prof.key_averages())

        if total_flops > 0:
            gflops = total_flops / 1e9
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            params_m = params / 1e6
            print(f"   - Trainable Parameters: {params_m:.8f} M")

            print(f"Model Complexity (Profiler):")
            print(f"   - GFLOPs: {gflops:.8f}")
            print(f"   - Trainable Parameters: {params_m:.8f} M")
            epoch_results.append(f"Model Complexity (Profiler): GFLOPs: {gflops:.8f}, Trainable Parameters: {params_m:.8f} M")
            
        else:
            # If this still fails, fvcore is the best alternative
            print("Profiler returned 0 FLOPs. Consider using the 'fvcore' method instead for a theoretical count.")

    except Exception as e:
        print(f"Could not calculate GFLOPs with PyTorch Profiler: {e}")
    # =====================================================================
    
    # Training Loop
    epoch_times = [] # Average Epoch Time 
    
    max_accuracy = 0.0 
    max_epoch = 0
    
    for epoch in range(args.num_epochs):
        # Model Training
        model.train() 
        train_running_loss = 0.0
        test_running_loss = 0.0
        epoch_result = ""
        
        start_time = time.time()

        train_top1_5 = [0, 0]
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # use mixed precision training
            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:    
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()            

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            train_top1_5[0] += top1.item()
            train_top1_5[1] += top5.item()
            train_running_loss += loss.item()

        train_top1_5[0] /= len(train_loader)
        train_top1_5[1] /= len(train_loader)
        end_time = time.time()
        epoch_result += f"[Epoch {epoch+1:03d}] Time: {end_time - start_time:.4f}s | [Train] Loss: {train_running_loss/len(train_loader):.8f} Accuracy: Top1: {train_top1_5[0]:.4f}%, Top5: {train_top1_5[1]:.4f}% | "
        epoch_times.append(end_time - start_time)
        
        # Model Evaluation 
        model.eval()
        test_top1_5 = [0, 0]
        with torch.no_grad():
            for images, labels in test_loader: 
                images, labels = images.to(device), labels.to(device)
                if args.use_amp:
                    with torch.cuda.amp.autocast():
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
        epoch_result += f"[Test] Loss: {test_running_loss/len(test_loader):.8f} Accuracy: Top1: {test_top1_5[0]:.4f}%, Top5: {test_top1_5[1]:.4f}%"
        print(epoch_result)
        epoch_results.append(epoch_result)
        
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
        

def accuracy(output, target, topk=(1,)):
    """Computes the top-1 and top-5 accuracy of the model."""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk] # [72.5, 91.3] - [top1, top5]