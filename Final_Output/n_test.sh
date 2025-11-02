#!/bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p arm --gres=shard:4
#SBATCH --cpus-per-task=12
#SBATCH --job-name=VIT-GH-NTEST
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention

# Setup conda
source ~/.bashrc
conda activate mingi-arm

### N-Test for CVPR paper 

# Configuration
DATASETS=("cifar10" "cifar100")
K_VALUES=("9")  
BLOCKS=("ConvNNAttention")
SAMPLING_TYPE=("random" "spatial")
N_SAMPLES=("16" "32" "48" "64" "80" "96" "112" "128" "144" "160" "176" "192")
LR="1e-4"                                         

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BLOCKS[@]} * ${#K_VALUES[@]} * ${#N_SAMPLES[@]} * ${#SAMPLING_TYPE[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "N-Test Configuration"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Datasets: ${DATASETS[@]}"
echo "Blocks: ${BLOCKS[@]}"
echo "K values: ${K_VALUES[@]}"
echo "N samples: ${N_SAMPLES[@]}"
echo "Sampling types: ${SAMPLING_TYPE[@]}"
echo "Learning rate: $LR"
echo "=========================================="
echo ""

# Main loop
for dataset in "${DATASETS[@]}"; do
    for block in "${BLOCKS[@]}"; do
        for k in "${K_VALUES[@]}"; do
            for n_samples in "${N_SAMPLES[@]}"; do
                for sampling_type in "${SAMPLING_TYPE[@]}"; do

                    COUNT=$((COUNT + 1))
                
                    # Create output directory
                    output_dir="./Final_Output/N_test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/${block}_K${k}_N${n_samples}_${sampling_type}_s42"
                    
                    echo "[$COUNT/$TOTAL] Dataset=$dataset | K=$k | Block=$block | N_Samples=$n_samples | Sampling_Type=$sampling_type"
                    echo "Output: $output_dir"
                    
                    # Single python call with padding set conditionally
                    python main.py \
                        --layer $block \
                        --patch_size 16 \ 
                        --num_layers 12 \ 
                        --num_heads 3 \ 
                        --d_hidden 192 \
                        --d_mlp 768 \
                        --dropout 0.1 \
                        --attention_dropout 0.1 \
                        --convolution_type depthwise \
                        --softmax_topk_val \
                        --K $k \
                        --sampling_type $sampling_type \
                        --num_samples $n_samples \ 
                        --magnitude_type matmul \ 
                        --dataset $dataset \
                        --resize 224 \
                        --batch_size 256 \
                        --num_epochs 150 \
                        --criterion CrossEntropy \
                        --optimizer adamw \
                        --weight_decay 1e-2 \
                        --lr $LR \
                        --clip_grad_norm 1.0 \
                        --scheduler none \ 
                        --seed 42 \
                        --device cuda \ 
                        --output_dir $output_dir
                    
                    # Check if experiment succeeded
                    if [ $? -eq 0 ]; then
                        echo "✓ Experiment $COUNT succeeded"
                    else
                        echo "✗ Experiment $COUNT failed"
                        FAILED=$((FAILED + 1))
                    fi
                    echo ""
                    
                
                done
            done
        done
    done
done




echo "=========================================="
echo "N-Test Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="