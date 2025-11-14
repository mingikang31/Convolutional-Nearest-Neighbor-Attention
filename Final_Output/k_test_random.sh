#!/bin/bash 

### Conv-Test for CVPR paper 

cd /home/exouser/Convolutional-Nearest-Neighbor-Attention/
# Configuration
DATASETS=("cifar100")
K_VALUES=("3" "4" "5" "6")  
BLOCKS=("ConvNNAttention") #"KvtAttention")
LR="1e-4"                                         

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BLOCKS[@]} * ${#K_VALUES[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "K-Test for Random Sampling Configuration"
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
            output_dir="./Final_Output/K_test_correct_random/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/${block}_Random_K${k}_NS32_s42"
            
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
                --sampling_type random \
                --num_samples 32 \
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



python main.py \
    --layer KvtAttention \
    --patch_size 16 \
    --num_layers 12 \
    --num_heads 1 \
    --d_hidden 192 \
    --d_mlp 768 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --K 100 \
    --dataset cifar10 \
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
    --output_dir "./Final_Output/K_test/ViT-Tiny-CIFAR10/KvtAttention_K100_s42"


python main.py \
    --layer KvtAttention \
    --patch_size 16 \
    --num_layers 12 \
    --num_heads 1 \
    --d_hidden 192 \
    --d_mlp 768 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --K 100 \
    --dataset cifar100 \
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
    --output_dir "./Final_Output/K_test/ViT-Tiny-CIFAR100/KvtAttention_K100_s42"


echo "=========================================="
echo "K-Test for Random Sampling Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="