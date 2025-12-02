#!/bin/bash 

### LR-Test

cd /home/exouser/Convolutional-Nearest-Neighbor-Attention/

# Configuration
DATASETS=("cifar10" "cifar100")
K_VALUES=("9")  
BLOCKS=("BranchAttention")
CONV_TYPES=("depthwise")
LR=("1e-5" "1e-4" "1e-3")                                

# Counter for progress
TOTAL=$((${#DATASETS[@]} * ${#BLOCKS[@]} * ${#K_VALUES[@]} * ${#LR[@]}))
COUNT=0
FAILED=0

echo "=========================================="
echo "LR-Test Configuration"
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
            for lr in "${LR[@]}"; do
                for conv_type in "${CONV_TYPES[@]}"; do

                    COUNT=$((COUNT + 1))
                
                    # Create output directory
                    output_dir="./Final_Output/lr_test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')_${lr}/${block}_${conv_type}_K${k}_cosine_s42"
                    
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
                        --convolution_type $conv_type \
                        --softmax_topk_val \
                        --K $k \
                        --sampling_type all \
                        --num_samples -1 \
                        --magnitude_type cosine \
                        --dataset $dataset \
                        --resize 224 \
                        --batch_size 256 \
                        --num_epochs 150 \
                        --criterion CrossEntropy \
                        --optimizer adamw \
                        --weight_decay 1e-2 \
                        --lr $lr \
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
echo "lr-Test Complete!"
echo "=========================================="
echo "Total experiments: $TOTAL"
echo "Successful: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "=========================================="