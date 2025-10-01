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


### CIFAR10 Experiments
python main.py --layer Attention --dataset cifar10 --output_dir ./Output/Oct1-VIT-Tiny-Baseline/CIFAR10/Attention_NH1_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct1-VIT-Tiny-Baseline/CIFAR10/ConvNNAttention_All_K9_NH1_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

### CIFAR100 Experiments

python main.py --layer Attention --dataset cifar100 --output_dir ./Output/Oct1-VIT-Tiny-Baseline/CIFAR100/Attention_NH1_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention --K 9 --sampling_type all --dataset cifar100 --output_dir ./Output/Oct1-VIT-Tiny-Baseline/CIFAR100/ConvNNAttention_All_K9_NH1_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1
