#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=CVPR-VIT
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


# CIFAR10
python main.py --layer Attention --output_dir ./Output/Sanity_Oct19/CIFAR10-Cosine/Attention_s42 --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar10 --scheduler cosine

python main.py --layer Attention --output_dir ./Output/Final_Oct21/CIFAR100-Cosine/Attention_s42 --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar100 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 9 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR10-Cosine/ConvNNKvt_K9_s42_mult_softmax --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar10 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 9 --sampling_type all --output_dir ./Output/Final_Oct21/CIFAR100-Cosine/ConvNNKvt_K9_s42_mult_softmax --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar100 --scheduler cosine
