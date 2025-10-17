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

### Sanity Check Experiments ###
# python main.py --layer Attention --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Attention_K9_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 

# ConvNN-Attention (Same as KvTAttention)
python main.py --layer ConvNNAttention_Same_KVT --K 9 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K9_s42_cosine_nosoftmax --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer ConvNNAttention_Same_KVT --K 16 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K16_s42_cosine_nosoftmax --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 25 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K25_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 36 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K36_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 49 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K49_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 64 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K64_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 81 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K81_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 100 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K100_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 121 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K121_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# python main.py --layer ConvNNAttention_Same_KVT --K 197 --sampling_type all --dataset cifar10 --output_dir ./Output/Sanity_Oct16/ConvNNKvt_K197_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

# KvTAttention
python main.py --layer KvtAttention --K 9 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K9_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 

python main.py --layer KvtAttention --K 16 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K16_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1 

python main.py --layer KvtAttention --K 25 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K25_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer KvtAttention --K 36 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K36_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer KvtAttention --K 49 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K49_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer KvtAttention --K 64 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K64_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer KvtAttention --K 81 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K81_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer KvtAttention --K 100 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K100_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer KvtAttention --K 121 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K121_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

python main.py --layer KvtAttention --K 197 --dataset cifar10 --output_dir ./Output/Sanity_Oct16/Kvt_K197_s42 --num_epochs 50 --seed 42 --lr 1e-5 --lr_step 2 --lr_gamma 0.95 --num_heads 1

