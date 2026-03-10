#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=triton_test
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-a100

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention/

# Triton
python main.py \
   --layer ConvNNAttention-Triton \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --use_amp \
   --compile \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/triton-amp-compile

python main.py \
   --layer ConvNNAttention-Triton \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --use_amp \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/triton-amp

python main.py \
   --layer ConvNNAttention-Triton \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --compile \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/triton-compile

python main.py \
   --layer ConvNNAttention-Triton \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/triton-none

# ConvNN Original
python main.py \
   --layer ConvNNAttention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --use_amp \
   --compile \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/original-amp-compile

python main.py \
   --layer ConvNNAttention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --use_amp \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/original-amp

python main.py \
   --layer ConvNNAttention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --compile \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/original-compile

python main.py \
   --layer ConvNNAttention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/original-none

# Attention
python main.py \
   --layer Attention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --use_amp \
   --compile \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/attention-amp-compile

python main.py \
   --layer Attention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --use_amp \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/attention-amp

python main.py \
   --layer Attention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --compile \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/attention-compile

python main.py \
   --layer Attention \
   --model vit-tiny \
   --num_heads 3 \
   --num_epochs 5 \
   --device cuda \
   --output_dir ./CUDA/Gemini/speed_dir/attention-none
