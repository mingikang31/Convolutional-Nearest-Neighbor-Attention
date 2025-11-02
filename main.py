"""Main File for VIT model"""

import argparse 
from pathlib import Path 
import os

import torch

# Datasets 
from dataset import ImageNet, CIFAR10, CIFAR100
from train_eval import Train_Eval

# Models
from vit import ViT

# Utils
from utils import write_to_file, set_seed

"""
VIT-Tiny Configuration 
patch_size: 16
num_layers:12
num_heads: 3
d_hidden: 192
d_mlp: 768
"""

def args_parser():
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor Attention training and evaluation", add_help=False) 
    
    # Model Arguments
    parser.add_argument("--layer", type=str, default="Attention", choices=["Attention", "ConvNNAttention", "KvtAttention", "LocalAttention", "NeighborhoodAttention", "BranchConv", "BranchAttention"], help="Layer to use for training and evaluation")

    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for Attention Models")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers in the model")   
    parser.add_argument("--num_heads", type=int, default=3, help="Number of heads for Attention Models")

    # Model Dimension Arguments
    parser.add_argument("--d_hidden", type=int, default=192, help="Hidden dimension for the model")
    parser.add_argument("--d_mlp", type=int, default=768, help="MLP dimension for the model")

    # Dropout Arguments
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--attention_dropout", type=float, default=0.1, help="Attention dropout rate for the model")    
    
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

    
    # Data Arguments
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", 'imagenet'], help="Dataset to use for training and evaluation")
    parser.add_argument("--resize", type=int, default=224, help="Resize images to 224x224 for Attention Models")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.set_defaults(augment=False)
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of Gaussian noise to add to the data")
    parser.add_argument("--data_path", type=str, default="./Data", help="Path to the dataset")
        
    # Training Arguments
    parser.add_argument("--use_compiled", action="store_true", help="Use compiled model for training and evaluation")
    parser.set_defaults(use_compiled=False)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "reduce-memory", "reduce-overhead", "max-autotune"], help="Compilation mode for torch.compile")

    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value")
    
    
    # Loss Function Arguments
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")
    
    # Optimizer Arguments 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer') # Only for SGD
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='Weight decay for optimizer') # For Adam & Adamw
    
    # Learning Rate Arguments
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler') # Only for StepLR
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for learning rate scheduler') # Only for StepLR
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau', 'none'], help='Learning rate scheduler')
    
    # Device Arguments
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"], help="Device to use for training and evaluation")
    parser.add_argument('--seed', default=0, type=int)
    
    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/VIT/VIT_Attention", help="Directory to save the output files")
    
    # Test Arguments
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.set_defaults(test_only=False)
    
    return parser
    
def main(args):
    
    # Dataset 
    if args.dataset == "cifar10":
        dataset = CIFAR10(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "cifar100":
        dataset = CIFAR100(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size 
    elif args.dataset == "imagenet":
        dataset = ImageNet(args)
        args.num_classes = dataset.num_classes 
        args.img_size = dataset.img_size
    else:
        raise ValueError("Dataset not supported")
    
    model = ViT(args)
    print(f"Model: {model.name}")

    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params

    
    if args.test_only:
        ex = torch.Tensor(3, 3, 224, 224).to(args.device)
        out = model(ex)
        print(f"Output shape: {out.shape}")
        print("Testing Complete")
    else:
        # Check if the output directory exists, if not create it
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # Set the seed for reproducibility
        set_seed(args.seed)
        
        # Training Modules 
        train_eval_results = Train_Eval(args, 
                                    model, 
                                    dataset.train_loader, 
                                    dataset.test_loader
                                    )
        
        # Storing Results in output directory 
        write_to_file(os.path.join(args.output_dir, "args.txt"), args)
        write_to_file(os.path.join(args.output_dir, "model.txt"), model)
        write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Convolutional Nearest Neighbor training and evaluation", parents=[args_parser()])
    args = parser.parse_args()

    main(args)
