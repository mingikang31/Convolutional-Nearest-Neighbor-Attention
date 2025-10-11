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

### CIFAR100 Experiments

# Attention Baseline

python main.py --layer Attention --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/Attention_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer Attention --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/Attention_s42_NH3 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 3


# ConvNN Attention Experiments
python main.py --layer ConvNNAttention --K 4 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_All_K4_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_All_K9_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 16 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_All_K16_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 25 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_All_K25_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1


### Depthwise Separable ConvNN Attention Experiments
python main.py --layer ConvNNAttention_Depthwise --K 4 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Depthwise_All_K4_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Depthwise_All_K9_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 16 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Depthwise_All_K16_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 25 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Depthwise_All_K25_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95


# ConvNN ConvNNAttention_Modified
python main.py --layer ConvNNAttention_Modified --K 4 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Modified_All_K4_s42_NH3 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 3

python main.py --layer ConvNNAttention_Modified --K 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Modified_All_K9_s42_NH3 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 3

python main.py --layer ConvNNAttention_Modified --K 16 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Modified_All_K16_s42_NH3 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 3

python main.py --layer ConvNNAttention_Modified --K 25 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/ConvNNAttention_Modified_All_K25_s42_NH3 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 3

# BranchingConv
python main.py --layer BranchConv --K 4 --kernel_size 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchConv_ConvNNAttention_All_K4_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

python main.py --layer BranchConv --K 9 --kernel_size 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchConv_ConvNNAttention_All_K9_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

python main.py --layer BranchConv --K 16 --kernel_size 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchConv_ConvNNAttention_All_K16_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

python main.py --layer BranchConv --K 25 --kernel_size 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchConv_ConvNNAttention_All_K25_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

# BranchingAttention
python main.py --layer BranchAttention --K 4 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchAttention_ConvNNAttention_All_K4_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

python main.py --layer BranchAttention --K 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchAttention_ConvNNAttention_All_K9_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

python main.py --layer BranchAttention --K 16 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchAttention_ConvNNAttention_All_K16_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1

python main.py --layer BranchAttention --K 25 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct11-VIT-Tiny-Sanity/CIFAR100/BranchAttention_ConvNNAttention_All_K25_br0500_s42_NH1 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --branch_ratio 0.5 --num_heads 1
