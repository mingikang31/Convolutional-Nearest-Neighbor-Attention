"""Main File for ZiLU Activation Function for Training GPT2 Models"""

import argparse 
from pathlib import Path 
import os 
import torch 

# Models 
from gpt2 import GPT2
from dataset import WikiText103
from train_eval import Train_Eval_GPT
from utils import write_to_file, set_seed 

"""Default Model: GPT2-small"""

"""
Parameters Layers d_model
Small - 117M 12 768
Medium - 345M 24 1024
Large - 762M 36 1280
XLarge - 1542M 48 1600
"""



def args_parser():
    parser = argparse.ArgumentParser(description="ZiLU Activation Function Experiments for GPT2 Models")

    # Model Args
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size for GPT2 model')
    parser.add_argument('--max_seq_length', type=int, default=1024, help='Maximum sequence length for GPT2 model')
    parser.add_argument('--embedding_dim', type=int, default=768, help='Embedding dimension for GPT2 model')
    parser.add_argument('--num_attention_heads', type=int, default=12, help='Number of attention heads for GPT2 model')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers for GPT2 model')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout for model')

    # ConvNN/Attention Layer Arguments
    parser.add_argument("--layer", type=str, default="Attention", choices=["Attention", "ConvNNAttention", "FastConvNNAttention", "EfficientConvNNAttention", "KvtAttention", "LocalAttention", "NeighborhoodAttention", "SparseAttention"], help="Layer to use for training and evaluation")

    # Additional Layer Arguments for ConvNN
    parser.add_argument("--convolution_type", type=str, default="depthwise", choices=["standard", "depthwise", "depthwise-separable"], help="Convolution type for ConvNN Layers")
    parser.add_argument("--K", type=int, default=9, help="K-nearest neighbor for ConvNN Layer")
    parser.add_argument("--sampling_type", type=str, default="all", choices=["all"], help="Sampling type for ConvNN Models")

    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples for ConvNN Layer, -1 for all samples")
    parser.add_argument("--sample_padding", type=int, default=0, help="Padding for spatial sampling in ConvNN Models")


    # Arguments for Data 
    parser.add_argument("--dataset", type=str, default="wikitext103", choices=["wikitext103"], help="Dataset to use for training and evaluation")
    parser.add_argument("--data_path", type=str, default="/mnt/research/j.farias/mkang2/Datasets", help="Path to the dataset")

    # Training Arguments
    parser.add_argument("--compile", action="store_true", help="Use compiled model for training and evaluation")
    parser.set_defaults(compile=False)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "reduce-memory", "reduce-overhead", "max-autotune"], help="Compilation mode for torch.compile")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
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
    
    # Optimizer Arguments
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for optimizer') # For Adam & Adamw

    # Learning Rate Arguments 
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate for the optimizer")
    parser.add_argument('--scheduler', type=str, default='linear', choices=['step', 'cosine', 'plateau', 'linear'], help='Learning rate scheduler type')
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler') # Only for StepLR
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Gamma for StepLR scheduler') # Only for StepLR

    # Device Arguments 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training and evaluation')
    parser.add_argument('--seed', default=0, type=int)

    # Output Arguments 
    parser.add_argument("--output_dir", type=str, default="./Output/GPT2", help="Directory to save the output files")

    # Test Arguments
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.set_defaults(test_only=False)

    # Distributed Data Parallel (DDP) 
    parser.add_argument("--ddp", action="store_true", help="Use Distributed Data Parallel (DDP) for training")
    parser.set_defaults(ddp=False)
    parser.add_argument("--ddp_batch_size", type=int, default=32, help="Batch size per GPU for DDP training")

    return parser 

def main(args):
    # Using TensorFloat-32 (TF32)
    try: 
        torch.set_float32_matmul_precision('high')
    except:
        print("Coult not use TensorFloat-32")
        
    # Dataset
    if args.dataset == "wikitext103":
        dataset = WikiText103(args)
    else:
        raise ValueError("Unsupported dataset. Currently only 'wikitext103' is supported.")
    
    # Initialize Model
    model = GPT2(
        args=args,
        vocab_size=args.vocab_size,
        max_seq_length=args.max_seq_length,
        embedding_dim=args.embedding_dim,
        num_attention_heads=args.num_attention_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=args.device
    )

    # Print Model Summary
    model.to(args.device) 
    print(f"Model: {model.name}")

    # Parameters
    total_params, trainable_params = model.parameter_count()
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    args.total_params = total_params
    args.trainable_params = trainable_params

    # Set Seed
    if args.test_only: 
        ex = torch.Tensor(1, args.max_seq_length).long().to(args.device)
        logits, loss = model(ex)
        print(f"Test Logits Shape: {logits.shape}")
        print("Testing Complete")
    else:
        # Check if the output directory exists, if not create it
        if args.output_dir: 
            Path(args.output_dir).mkdir(parents=True, exist_ok=True) 

        # Training Module
        train_eval_results = Train_Eval_GPT(
            args=args,
            model=model, 
            train_loader=dataset.train_loader, 
            test_loader=dataset.test_loader, 
            val_loader=dataset.val_loader
        )

        # Store Results
        write_to_file(os.path.join(args.output_dir, "args.txt"), args) 
        write_to_file(os.path.join(args.output_dir, "model.txt"), model) 
        write_to_file(os.path.join(args.output_dir, "train_eval_results.txt"), train_eval_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPT2 with ZiLU Activation Function", parents=[args_parser()], add_help=False)
    args = parser.parse_args()
    main(args) 