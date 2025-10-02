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
# num_heads: 3
# d_hidden: 192
# d_mlp: 768

# K = 9
# Rand & Spatial: N = 32

### CIFAR10 Experiments
python main.py --layer BranchConv --K 9 --kernel_size 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct1-VIT-Tiny-Baseline/CIFAR10/BranchConv_ConvNNAttention_All_K9_br0500_s42_KQV --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5

python main.py --layer BranchAttention --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct1-VIT-Tiny-Baseline/CIFAR10/BranchAttention_ConvNNAttention_All_K9_br0500_s42_KQV --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5


# TODO 
# 1. ConvNN Attention with K, Q, V Projection 
# 2. ConvNN Attention with V Projection Only 
# 3. ConvNN Attention with No Projections (K = Q = V = Input)

# 4. Branching with Conv2d + ConvNN Attention with K, Q, V Projection
# 5. Branching with Conv2d + ConvNN Attention with V Projection Only
# 6. Branching with Conv2d + ConvNN Attention with No Projections (K = Q = V = Input)

# 7. Branching with Attention + ConvNN Attention with K, Q, V Projection
# 8. Branching with Attention + ConvNN Attention with V Projection Only
# 9. Branching with Attention + ConvNN Attention with No Projections (K = Q = V = Input) 