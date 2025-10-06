#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=VIT_Tiny_sep23
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention

source activate mingi


# VIT-Tiny Configuration 
# patch_size: 16
# num_layers:12
# num_heads: 1
# d_hidden: 192
# d_mlp: 768

# K = 9
# Rand & Spatial: N = 32

### CIFAR10 Experiments

# python main.py --layer Attention --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/Attention_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1


python main.py --layer ConvNNAttention --K 4 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/ConvNNAttention_All_K4_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 16 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/ConvNNAttention_All_K16_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 25 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/ConvNNAttention_All_K25_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 36 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/ConvNNAttention_All_K36_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 49 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/ConvNNAttention_All_K49_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1


# python main.py --layer BranchConv --K 9 --kernel_size 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/BranchConv_ConvNNAttention_All_K9_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

# python main.py --layer BranchAttention --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10/BranchAttention_ConvNNAttention_All_K9_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

