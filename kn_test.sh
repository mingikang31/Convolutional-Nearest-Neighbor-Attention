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

# ### K-Test-CIFAR10
# K = 1
python main.py --layer ConvNNAttention --K 1 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_All_K1 --num_epochs 50

python main.py --layer ConvNNAttention --K 1 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Random_K1 --num_epochs 50

python main.py --layer ConvNNAttention --K 1 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Spatial_K1 --num_epochs 50

# K = 2
python main.py --layer ConvNNAttention --K 2 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_All_K2 --num_epochs 50

python main.py --layer ConvNNAttention --K 2 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Random_K2 --num_epochs 50

python main.py --layer ConvNNAttention --K 2 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Spatial_K2 --num_epochs 50

# K = 3
python main.py --layer ConvNNAttention --K 3 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_All_K3 --num_epochs 50

python main.py --layer ConvNNAttention --K 3 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Random_K3 --num_epochs 50

python main.py --layer ConvNNAttention --K 3 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Spatial_K3 --num_epochs 50

# K = 4
python main.py --layer ConvNNAttention --K 4 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_All_K4 --num_epochs 50

python main.py --layer ConvNNAttention --K 4 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Random_K4 --num_epochs 50

python main.py --layer ConvNNAttention --K 4 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Spatial_K4 --num_epochs 50

# K = 6
python main.py --layer ConvNNAttention --K 6 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_All_K6 --num_epochs 50

python main.py --layer ConvNNAttention --K 6 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Random_K6 --num_epochs 50

python main.py --layer ConvNNAttention --K 6 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Spatial_K6 --num_epochs 50


# K = 9
python main.py --layer ConvNNAttention --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_All_K9 --num_epochs 50

python main.py --layer ConvNNAttention --K 9 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Random_K9 --num_epochs 50

python main.py --layer ConvNNAttention --K 9 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Spatial_K9 --num_epochs 50


# K = 11
python main.py --layer ConvNNAttention --K 11 --sampling_type all --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_All_K11 --num_epochs 50

python main.py --layer ConvNNAttention --K 11 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Random_K11 --num_epochs 50

python main.py --layer ConvNNAttention --K 11 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/K-Test-CIFAR10/ConvNNAttention_Spatial_K11 --num_epochs 50


# ### N-Test-CIFAR10
# K = 4, N = 16
python main.py --layer ConvNNAttention --K 4 --sampling_type random --num_samples 16 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Random_K4_N16 --num_epochs 50

python main.py --layer ConvNNAttention --K 4 --sampling_type spatial --num_samples 16 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Spatial_K4_N16 --num_epochs 50

# K = 4, N = 32
python main.py --layer ConvNNAttention --K 4 --sampling_type random --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Random_K4_N32 --num_epochs 50

python main.py --layer ConvNNAttention --K 4 --sampling_type spatial --num_samples 32 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Spatial_K4_N32 --num_epochs 50

# K = 4, N = 48
python main.py --layer ConvNNAttention --K 4 --sampling_type random --num_samples 48 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Random_K4_N48 --num_epochs 50

python main.py --layer ConvNNAttention --K 4 --sampling_type spatial --num_samples 48 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Spatial_K4_N48 --num_epochs 50

# K = 4, N = 64
python main.py --layer ConvNNAttention --K 4 --sampling_type random --num_samples 64 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Random_K4_N64 --num_epochs 50

python main.py --layer ConvNNAttention --K 4 --sampling_type spatial --num_samples 64 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Spatial_K4_N64 --num_epochs 50

# K = 4, N = 128
python main.py --layer ConvNNAttention --K 4 --sampling_type random --num_samples 128 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Random_K4_N128 --num_epochs 50

python main.py --layer ConvNNAttention --K 4 --sampling_type spatial --num_samples 128 --dataset cifar10 --output_dir ./Output/VIT-Tiny/N-Test-CIFAR10/ConvNNAttention_Spatial_K4_N128 --num_epochs 50
