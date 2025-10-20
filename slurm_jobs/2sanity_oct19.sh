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

### Step scheduler ###
# CIFAR100 Step
python main.py --layer Attention --output_dir ./Output/Sanity_Oct19/CIFAR100-Step/Attention_K9_s42 --num_epochs 100 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 --dataset cifar100 --scheduler step

python main.py --layer ConvNNAttention_Same_KVT --K 9 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR100-Step/ConvNNKvt_K9_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 --dataset cifar100 --scheduler step

python main.py --layer ConvNNAttention_Same_KVT --K 16 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR100-Step/ConvNNKvt_K16_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 --dataset cifar100 --scheduler step

# CIFAR10 Step
python main.py --layer Attention --output_dir ./Output/Sanity_Oct19/CIFAR10-Step/Attention_K9_s42 --num_epochs 100 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 --dataset cifar10 --scheduler step

python main.py --layer ConvNNAttention_Same_KVT --K 9 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR10-Step/ConvNNKvt_K9_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 --dataset cifar10 --scheduler step

python main.py --layer ConvNNAttention_Same_KVT --K 16 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR10-Step/ConvNNKvt_K16_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 --dataset cifar10 --scheduler step




### Cosine scheduler ###
# CIFAR100 Cosine
python main.py --layer Attention --output_dir ./Output/Sanity_Oct19/CIFAR100-Cosine/Attention_K9_s42 --num_epochs 100 --seed 42 --lr 1e-5 --num_heads 1 --dataset cifar100 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 9 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR100-Cosine/ConvNNKvt_K9_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --num_heads 1 --dataset cifar100 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 16 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR100-Cosine/ConvNNKvt_K16_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --num_heads 1 --dataset cifar100 --scheduler cosine

# CIFAR10 Cosine
python main.py --layer Attention --output_dir ./Output/Sanity_Oct19/CIFAR10-Cosine/Attention_K9_s42 --num_epochs 100 --seed 42 --lr 1e-5 --num_heads 1 --dataset cifar10 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 9 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR10-Cosine/ConvNNKvt_K9_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --num_heads 1 --dataset cifar10 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 16 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR10-Cosine/ConvNNKvt_K16_s42_cosine_softmax --num_epochs 100 --seed 42 --lr 1e-5 --num_heads 1 --dataset cifar10 --scheduler cosine