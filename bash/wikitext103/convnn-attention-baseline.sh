#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=128G
#SBATCH -p arm --gres=shard:32
#SBATCH --cpus-per-task=48
#SBATCH --job-name=convnn_gpt2
#SBATCH --time=96:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate torch-gh200

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention


COUNT=0
FAILED=0

export TORCH_FLOAT32_MATMUL_PRECISION=high
LR="6e-4"
dataset="wikitext103"


# ConvNN Attention Baseline
output_dir="./Output/WikiText103/convnn_K4_s42"

python language_main.py \
    --vocab_size 50257 \
    --max_seq_length 1024 \
    --embedding_dim 768 \
    --num_attention_heads 12 \
    --num_layers 12 \
    --layer ConvNNAttention \
    --convolution_type depthwise \
    --K 4 \
    --sampling_type all \
    --dataset $dataset \
    --use_amp \
    --data_path ./Data \
    --batch_size 32 \
    --num_epochs 20 \
    --clip_grad_norm 1.0 \
    --optimizer adamw \
    --weight_decay 0.1 \
    --lr $LR \
    --scheduler linear \
    --device cuda \
    --seed 42 \
    --output_dir $output_dir \
    --num_workers 14 \
    --pin_memory

EXIT_CODE=$?
COUNT=$((COUNT + 1))
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi

# Regular Attention Baseline
output_dir="./Output/WikiText103/self-attention_baseline_s42"

python language_main.py \
    --vocab_size 50257 \
    --max_seq_length 1024 \
    --embedding_dim 768 \
    --num_attention_heads 12 \
    --num_layers 12 \
    --layer Attention \
    --dataset $dataset \
    --use_amp \
    --data_path ./Data \
    --batch_size 32 \
    --num_epochs 20 \
    --clip_grad_norm 1.0 \
    --optimizer adamw \
    --weight_decay 0.1 \
    --lr $LR \
    --scheduler linear \
    --device cuda \
    --seed 42 \
    --output_dir $output_dir \
    --num_workers 14 \
    --pin_memory


EXIT_CODE=$?
COUNT=$((COUNT + 1))
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment $COUNT succeeded"
else
    echo "✗ Experiment $COUNT failed"
    FAILED=$((FAILED + 1))
fi

echo "============================="
echo "Results: $((COUNT - FAILED))/$COUNT experiments succeeded"
if [ $FAILED -gt 0 ]; then
    echo "$FAILED experiment(s) failed"
fi