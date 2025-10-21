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

python main.py --layer ConvNNAttention_Same_KVT --K 6 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR10-Cosine/ConvNNKvt_K6_s42_mult_softmax --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar10 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 12 --sampling_type all --output_dir ./Output/Sanity_Oct19/CIFAR10-Cosine/ConvNNKvt_K12_s42_mult_softmax --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar10 --scheduler cosine


python main.py --layer ConvNNAttention_Same_KVT --K 6 --sampling_type all --output_dir ./Output/Final_Oct21/CIFAR100-Cosine/ConvNNKvt_K6_s42_mult_softmax --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar100 --scheduler cosine

python main.py --layer ConvNNAttention_Same_KVT --K 12 --sampling_type all --output_dir ./Output/Final_Oct21/CIFAR100-Cosine/ConvNNKvt_K12_s42_mult_softmax --num_epochs 200 --seed 42 --optimizer adamw --weight_decay 1e-2 --lr 1e-4 --num_heads 1 --dataset cifar100 --scheduler cosine
