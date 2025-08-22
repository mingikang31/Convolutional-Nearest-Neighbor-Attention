#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=default-exps
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
# No Coordinate Encoding

### CIFAR10 Experiments

python main.py --layer Attention --dataset cifar10 --output_dir ./Output/VIT-Tiny/CIFAR10/Attention

python main.py --layer ConvNNAttention --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/CIFAR10/ConvNNAttention_All 

python main.py --layer ConvNNAttention --K 9 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/CIFAR10/ConvNNAttention_Random

python main.py --layer ConvNNAttention --K 9 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/CIFAR10/ConvNNAttention_Spatial

python main.py --layer KvtAttention --K 9 --dataset cifar10 --output_dir ./Output/VIT-Tiny/CIFAR10/KvtAttention

python main.py --layer LocalAttention --dataset cifar10 --output_dir ./Output/VIT-Tiny/CIFAR10/LocalAttention

python main.py --layer NeighborhoodAttention --dataset cifar10 --output_dir ./Output/VIT-Tiny/CIFAR10/NeighborhoodAttention

### CIFAR100 Experiments

python main.py --layer Attention --dataset cifar100 --output_dir ./Output/VIT-Tiny/CIFAR100/Attention

python main.py --layer ConvNNAttention --K 9 --sampling_type all --dataset cifar100 --output_dir ./Output/VIT-Tiny/CIFAR100/ConvNNAttention_All 

python main.py --layer ConvNNAttention --K 9 --sampling_type random --num_samples 32 --dataset cifar100 --output_dir ./Output/VIT-Tiny/CIFAR100/ConvNNAttention_Random

python main.py --layer ConvNNAttention --K 9 --sampling_type spatial --num_samples 32 --dataset cifar100 --output_dir ./Output/VIT-Tiny/CIFAR100/ConvNNAttention_Spatial

python main.py --layer KvtAttention --K 9 --dataset cifar100 --output_dir ./Output/VIT-Tiny/CIFAR100/KvtAttention

python main.py --layer LocalAttention --dataset cifar100 --output_dir ./Output/VIT-Tiny/CIFAR100/LocalAttention

python main.py --layer NeighborhoodAttention --dataset cifar100 --output_dir ./Output/VIT-Tiny/CIFAR100/NeighborhoodAttention