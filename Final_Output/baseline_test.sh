#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p arm --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --job-name=VIT-GH-Baseline
#SBATCH --time=96:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention

# Setup conda
source ~/.bashrc
conda activate mingi-arm


# VIT-Tiny Configuration 
# patch_size: 16
# num_layers:12
# num_heads: 3
# d_hidden: 192
# d_mlp: 768

# Configuration 
DATASETS=("cifar10" "cifar100")
LR="1e-3"

# Main Loop 
for dataset in "${DATASETS[@]}"; do

    Attention Baseline
    output_dir1="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/Attention_s42"

    python main.py --layer Attention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
       --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
       --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir1

    # Local Attention Baseline 
    output_dir2="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/LocalAttention_s42"

    python main.py --layer LocalAttention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
       --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
       --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir2

    # # Neighborhood Attention
    # output_dir3="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/NeighborhoodAttention_s42"
    # python main.py --layer NeighborhoodAttention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --K 3 \
    #     --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
    #     --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir3

done

