#! /bin/bash 
#SBATCH --nodes=1 
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=a100
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out
#SBATCH --error=slurm_out/%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=mkang2@bowdoin.edu

source ~/.bashrc
conda activate mingi
cd /mnt/research/j.farias/mkang2/Convolutional-Nearest-Neighbor-Attention

# Configuration
DATASETS=("cifar10" "cifar100")
K_VALUES=("16" "25" "36")  
BLOCKS=("KvtAttention")
LR="1e-4"                                         

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BLOCKS[@]} * ${#K_VALUES[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "K-Test Configuration"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Datasets: ${DATASETS[@]}"
echo "Blocks: ${BLOCKS[@]}"
echo "K values: ${K_VALUES[@]}"
echo "Learning rate: $LR"
echo "=========================================="
echo ""

# Main loop
for dataset in "${DATASETS[@]}"; do
    for block in "${BLOCKS[@]}"; do
        for k in "${K_VALUES[@]}"; do

            COUNT=$((COUNT + 1))
        
            # Create output directory
            output_dir="./Final_Output/K_test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/${block}_K${k}_s42"
            
            echo "[$COUNT/$TOTAL] Dataset=$dataset | K=$k | Block=$block"
            echo "Output: $output_dir"
            
            # Single python call with padding set conditionally
            python main.py \
                --layer $block \
                --patch_size 16 \
                --num_layers 12 \
                --num_heads 1 \
                --d_hidden 192 \
                --d_mlp 768 \
                --dropout 0.1 \
                --attention_dropout 0.1 \
                --convolution_type depthwise \
                --softmax_topk_val \
                --K $k \
                --sampling_type all \
                --num_samples -1 \
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



# python main.py \
#     --layer KvtAttention \
#     --patch_size 16 \ 
#     --num_layers 12 \ 
#     --num_heads 3 \ 
#     --d_hidden 192 \
#     --d_mlp 768 \
#     --dropout 0.1 \
#     --attention_dropout 0.1 \
#     --K 100 \
#     --dataset cifar10 \
#     --resize 224 \
#     --batch_size 256 \
#     --num_epochs 150 \
#     --criterion CrossEntropy \
#     --optimizer adamw \
#     --weight_decay 1e-2 \
#     --lr $LR \
#     --clip_grad_norm 1.0 \
#     --scheduler none \ 
#     --seed 42 \
#     --device cuda \ 
#     --output_dir "./Final_Output/K_test/ViT-Tiny-CIFAR10/KvtAttention_K100_s42"


# python main.py \
#     --layer KvtAttention \
#     --patch_size 16 \ 
#     --num_layers 12 \ 
#     --num_heads 3 \ 
#     --d_hidden 192 \
#     --d_mlp 768 \
#     --dropout 0.1 \
#     --attention_dropout 0.1 \
#     --K 100 \
#     --dataset cifar100 \
#     --resize 224 \
#     --batch_size 256 \
#     --num_epochs 150 \
#     --criterion CrossEntropy \
#     --optimizer adamw \
#     --weight_decay 1e-2 \
#     --lr $LR \
#     --clip_grad_norm 1.0 \
#     --scheduler none \ 
#     --seed 42 \
#     --device cuda \ 
#     --output_dir "./Final_Output/K_test/ViT-Tiny-CIFAR100/KvtAttention_K100_s42"


echo "=========================================="
echo "K-Test Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="