#! /bin/bash 
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

# Create output directory

original_dir="./CUDA/Output/ViT-Tiny-ConvNNAttention-Original"
triton_dir="./CUDA/Output/ViT-Tiny-ConvNNAttention-Triton"

python main.py \
   --layer ConvNNAttention-Triton \
   --patch_size 16 \
   --num_layers 12 \
   --num_heads 1 \
   --d_hidden 192 \
   --d_mlp 768 \
   --dropout 0.1 \
   --attention_dropout 0.1 \
   --convolution_type depthwise \
   --K 9 \
   --sampling_type all \
   --num_samples -1 \
   --dataset cifar100 \
   --resize 224 \
   --data_path ./Data \
   --batch_size 128 \
   --num_epochs 200 \
   --use_amp \
   --clip_grad_norm 1.0 \
   --criterion CrossEntropy \
   --optimizer adamw \
   --weight_decay 1e-2 \
   --lr 1e-3 \
   --scheduler cosine \
   --device cuda \
   --seed 42 \
   --output_dir $triton_dir \
   --num_workers 12 \
   --pin_memory

python main.py \
   --layer ConvNNAttention \
   --patch_size 16 \
   --num_layers 12 \
   --num_heads 1 \
   --d_hidden 192 \
   --d_mlp 768 \
   --dropout 0.1 \
   --attention_dropout 0.1 \
   --convolution_type depthwise \
   --K 9 \
   --sampling_type all \
   --num_samples -1 \
   --dataset cifar100 \
   --resize 224 \
   --data_path ./Data \
   --batch_size 128 \
   --num_epochs 200 \
   --use_amp \
   --clip_grad_norm 1.0 \
   --criterion CrossEntropy \
   --optimizer adamw \
   --weight_decay 1e-2 \
   --lr 1e-3 \
   --scheduler cosine \
   --device cuda \
   --seed 42 \
   --output_dir $original_dir \
   --num_workers 12 \
   --pin_memory
