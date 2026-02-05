"""Main File for Convolutional Nearest Neighbors Attention Training and Evaluation"""

import argparse 
from pathlib import Path
import os 

# Torch
import torch 
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist 

# Datasets & Eval
from dataset import CIFAR10, CIFAR100, ImageNet1K, mixup_fn
from train_eval import Train_Eval, Train_Eval_ImageNet, setup_distributed, cleanup_distributed

# Models 
from vit import ViT, ViT_DropPath

# Utilities 
from utils import write_to_file, set_seed

# Warnings and Logging 
import warnings 
warnings.filterwarnings("ignore")
import logging 
logging.getLogger().setLevel(logging.ERROR)


"""
VIT-Tiny Configuration 
patch_size: 16
num_layers:12
num_heads: 3
d_hidden: 192
d_mlp: 768
"""

"""
Drop Path Values 
DeiT-Small = 0.1 
DeiT-Base = 0.1
Swin-tiny = 0.2
Swin-small = 0.3
Swin-base = 0.5

"""

def args_parser():
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor Attention training and evaluation", add_help=False) 

    parser.add_argument("--model", type=str, default="vit-tiny", 
                        choices=["vit-tiny", "vit-small", "vit-base", "vit-large", "vit-huge", 

                                 ], help="Model architecture to use for training and evaluation"
                        )
    
    # Model Arguments
    parser.add_argument("--layer", type=str, default="Attention", choices=["Attention", "ConvNNAttention", "KvtAttention", "LocalAttention", "NeighborhoodAttention", "SparseAttention", "BranchConv", "BranchAttention"], help="Layer to use for training and evaluation")

    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for Attention Models")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the model")   
    parser.add_argument("--num_heads", type=int, default=3, help="Number of heads for Attention Models")

    # Model Dimension Arguments
    parser.add_argument("--d_hidden", type=int, default=192, help="Hidden dimension for the model")
    parser.add_argument("--d_mlp", type=int, default=768, help="MLP dimension for the model")

    # Dropout Arguments
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Attention dropout rate for the model")    
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="DropPath rate for the model")
    
    # Additional Layer Arguments for ConvNN
    parser.add_argument("--convolution_type", type=str, default="depthwise", choices=["standard", "depthwise", "depthwise-separable"], help="Convolution type for ConvNN Layers")
    parser.add_argument("--softmax_topk_val", action="store_true", help="Use top-k values for softmax computation in Attention Models")
    parser.set_defaults(softmax_topk_val=True)
    parser.add_argument("--K", type=int, default=9, help="K-nearest neighbor for ConvNN Layer")
    parser.add_argument("--sampling_type", type=str, default="all", choices=["all", "random", "spatial"], help="Sampling type for ConvNN Models")

    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples for ConvNN Layer, -1 for all samples")
    parser.add_argument("--sample_padding", type=int, default=0, help="Padding for spatial sampling in ConvNN Models")
    parser.add_argument("--magnitude_type", type=str, default="matmul", choices=["cosine", "euclidean", "matmul"], help="Magnitude type for ConvNN Models")
    parser.add_argument("--coordinate_encoding", action="store_true", help="Use coordinate encoding in ConvNN Models")
    parser.set_defaults(coordinate_encoding=False)    
    parser.add_argument("--branch_ratio", type=float, default=0.5, help="Branch ratio for ConvNN Models")

    # Additional Layer Arguments for Conv1d
    parser.add_argument("--kernel_size", type=int, default=9, help="Kernel size for Conv1d Layer")

    # Additional Layer Arguments for Sparse Attention
    parser.add_argument("--sparse_mode", type=str, default="all", choices=["all", "local", "strided"], help="Sparsity mode for Sparse Attention Models")
    parser.add_argument("--sparse_context_window", type=int, default=128, help="Context window for Sparse Attention Models")

    
    # Data Arguments
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", 'imagenet1k'], help="Dataset to use for training and evaluation")
    parser.add_argument("--resize", type=int, default=224, help="Resize images to 224x224 for Attention Models")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.set_defaults(augment=False)
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of Gaussian noise to add to the data")
    parser.add_argument("--data_path", type=str, default="./Data", help="Path to the dataset")
        
    # Training Arguments
    parser.add_argument("--compile", action="store_true", help="Use compiled model for training and evaluation")
    parser.set_defaults(compile=False)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "reduce-memory", "reduce-overhead", "max-autotune"], help="Compilation mode for torch.compile")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value")

    # Data Loader Specific 
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--persistent_workers", action="store_true", help="Use persistent workers for data loading")
    parser.set_defaults(persistent_workers=False)
    parser.add_argument("--prefetch_factor", type=int, default=None, help="Prefetch factor for data loading")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for data loading")
    parser.set_defaults(pin_memory=False)    
    
    # Loss Function Arguments
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer') # Only for SGD
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for optimizer') # For Adam & Adamw
    
    # Learning Rate Arguments
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler') # Only for StepLR
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for learning rate scheduler') # Only for StepLR
    parser.add_argument('--scheduler', type=str, default='none', choices=['step', 'cosine', 'plateau', 'none'], help='Learning rate scheduler')
    
    # Device Arguments
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"], help="Device to use for training and evaluation")
    parser.add_argument('--seed', default=0, type=int)
    
    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/VIT/VIT_Attention", help="Directory to save the output files")
    
    # Test Arguments
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.set_defaults(test_only=False)

    # Distributed Data Parallel (DDP)
    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel (DDP) for training")
    parser.set_defaults(ddp=False)
    parser.add_argument("--ddp_batch_size", type=int, default=128, help="Batch size per GPU for DDP training")
    
    return parser
    
def main(args):
    # Using TensorFloat-32 (TF32) 
    try: 
        torch.set_float32_matmul_precision('high')
    except: 
        print("Could not use TensorFloat-32")

    # DDP Setup 
    local_rank = setup_distributed()
    if dist.is_initialized() and dist.is_available():
        args.ddp = True 
        print(f"DDP is initialized. Local Rank: {local_rank}, World Size: {dist.get_world_size()}")    

    # Dataset 
    if args.dataset == "cifar10":
        dataset = CIFAR10(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "cifar100":
        dataset = CIFAR100(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "imagenet1k":
        args.batch_size = 1024 # Standard Batch Size for ImageNet-1K
        args.augment = True
        dataset = ImageNet1K(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size
    else:
        raise ValueError("Dataset not supported")


    if args.model == "vit-tiny":
        args.patch_size = 16
        args.num_layers = 12
        args.num_heads = 3
        args.d_hidden = 192
        args.d_mlp = 768
        
    elif args.model == "vit-small":
        args.patch_size = 16
        args.num_layers = 12
        args.num_heads = 6
        args.d_hidden = 384
        args.d_mlp = 1536
        
    elif args.model == "vit-base":
        args.patch_size = 16
        args.num_layers = 12
        args.num_heads = 12
        args.d_hidden = 768
        args.d_mlp = 3072
        
    elif args.model == "vit-large":
        args.patch_size = 16
        args.num_layers = 24
        args.num_heads = 16
        args.d_hidden = 1024
        args.d_mlp = 4096
        
    elif args.model == "vit-huge":
        args.patch_size = 14
        args.num_layers = 32
        args.num_heads = 16
        args.d_hidden = 1280
        args.d_mlp = 5120
    else:
        raise ValueError("Model not supported")

    if args.dataset == "imagenet1k":
        model = ViT_DropPath(args)
    else:
        model = ViT(args)

    model.to(args.device, memory_format=torch.channels_last) # For UserWarning: Grad strides do not match view strides 
    
    print(f"Model: {model.name}")

    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params

    # Distributed Data Parallel (DDP)
    if args.ddp and dist.is_available() and dist.is_initialized():
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # only necessary if using smaller gpu and smaller batch size (ex. three rtx3080 with batch size 32 each)
        model = DDP(model, device_ids=[local_rank])

    if args.test_only:
        ex = torch.Tensor(3, 3, 224, 224).to(args.device)
        out = model(ex)
        print(f"Output shape: {out.shape}")
        print("Testing Complete")
    else:
        # Check if the output directory exists, if not create it
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)


        # Training Module 
        if args.dataset in ["cifar10", "cifar100"]:
            set_seed(args.seed)
            train_eval_results = Train_Eval(args, 
                                        model, 
                                        dataset.train_loader, 
                                        dataset.test_loader
                                        )
        elif args.dataset == "imagenet1k":
            train_eval_results = Train_Eval_ImageNet(
                                        args, 
                                        model, 
                                        dataset.train_loader, 
                                        dataset.test_loader,
                                        train_sampler=dataset.train_sampler,
                                        mixup_fn=mixup_fn, 
                                        rank=local_rank
                                        )            

        # Cleanup DDP 
        cleanup_distributed()
            
        # Storing Results in output directory 
        write_to_file(os.path.join(args.output_dir, "args.txt"), args)
        write_to_file(os.path.join(args.output_dir, "model.txt"), model)
        write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbors Attention", parents=[args_parser()])
    args = parser.parse_args()
    main(args)