#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=Oct7_VIT
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

python main.py --layer Attention --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10-newLR/Attention_s42_NH1 --num_epochs 100 --seed 42 --batch_size 128 --lr 2e-4 --optimizer adamw --weight_decay 0.05 --scheduler cosine --dropout 0.1 --attention_dropout 0.1 --clip_grad_norm 1.0 --num_heads 1

python main.py --layer ConvNNAttention --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10-newLR/ConvNNAttention_All_K9_s42_NH1 --num_epochs 100 --seed 42 --batch_size 128 --lr 2e-4 --optimizer adamw --weight_decay 0.05 --scheduler cosine --dropout 0.1 --attention_dropout 0.1 --clip_grad_norm 1.0 --num_heads 1

python main.py --layer ConvNNAttention --K 25 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10-newLR/ConvNNAttention_All_K25_s42_NH1 --num_epochs 100 --seed 42 --batch_size 128 --lr 2e-4 --optimizer adamw --weight_decay 0.05 --scheduler cosine --dropout 0.1 --attention_dropout 0.1 --clip_grad_norm 1.0 --num_heads 1

python main.py --layer BranchConv --K 9 --kernel_size 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10-newLR/BranchConv_ConvNNAttention_All_K9_br0500_s42_NH1 --num_epochs 100 --seed 42 --batch_size 128 --lr 2e-4 --optimizer adamw --weight_decay 0.05 --scheduler cosine --dropout 0.1 --attention_dropout 0.1 --clip_grad_norm 1.0 --num_heads 1

python main.py --layer BranchAttention --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Oct6-VIT-Tiny-Sanity/CIFAR10-newLR/BranchAttention_ConvNNAttention_All_K9_br0500_s42_NH1 --num_epochs 100 --seed 42 --batch_size 128 --lr 2e-4 --optimizer adamw --weight_decay 0.05 --scheduler cosine --dropout 0.1 --attention_dropout 0.1 --clip_grad_norm 1.0 --num_heads 1
