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
python main.py --layer ConvNNAttention_Depthwise --K 4 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K4_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K9_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 16 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K16_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 25 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K25_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 36 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K36_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 49 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K49_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 64 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K64_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 81 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K81_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

python main.py --layer ConvNNAttention_Depthwise --K 100 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct9-VIT-Tiny-Depthwise/CIFAR10/ConvNNAttention_Depthwise_All_K100_s42_noheads --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95

