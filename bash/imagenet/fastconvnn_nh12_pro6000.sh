#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=480G
#SBATCH -p mixed --gres=gpu:pro6000:1
#SBATCH --cpus-per-gpu=80
#SBATCH --job-name=convnn-imgnet
#SBATCH --time=720:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-pro6000

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention

python main.py \
    --model vit-base \
    --layer FastConvNNAttention \
    --K 100 \
    --convolution_type depthwise \
    --sampling_type all \
    --num_samples -1 \
    --num_heads 12 \
    --dataset imagenet1k \
    --data_path /mnt/research/j.farias/mkang2/Datasets \
    --use_amp \
    --batch_size 256 \
    --device cuda \
    --output_dir ./Output/ImageNet1K/ViT-Base-FastConvNNAttention_NH12_All_K100/ \
    --num_workers 16 \
    --pin_memory