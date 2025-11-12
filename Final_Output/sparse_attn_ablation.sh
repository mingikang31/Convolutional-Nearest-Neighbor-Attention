#!/bin/bash 

### Conv-Test for CVPR paper 

cd /home/exouser/Convolutional-Nearest-Neighbor-Attention/

# VIT-Tiny Configuration 
# patch_size: 16
# num_layers:12
# num_heads: 3
# d_hidden: 192
# d_mlp: 768

# Configuration 
DATASETS=("cifar10" "cifar100")
LR="1e-4"
CONTEXT_WINDOWS=(32 64 96 128 196)  

for dataset in "${DATASETS[@]}"; do
    # Test different context windows
    for cw in "${CONTEXT_WINDOWS[@]}"; do

        echo "=========================================="
        echo "Running Local mode with CW=${cw} on ${dataset}"
        echo "=========================================="
        
        # ... rest of script
        output_dir="./Final_Output/Baseline_Test_Sparse/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/SparseAttention_Local_NH1_CW${cw}_s42"

        python main.py --layer SparseAttention --patch_size 16 --num_layers 12 --num_heads 1 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
            --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
            --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir --sparse_mode "local" --sparse_context_window $cw


        echo "=========================================="
        echo "Running Strided mode with CW=${cw} on ${dataset}"
        echo "=========================================="
        
        # ... rest of script
        output_dir2="./Final_Output/Baseline_Test_Sparse/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/SparseAttention_Strided_NH1_CW${cw}_s42"

        python main.py --layer SparseAttention --patch_size 16 --num_layers 12 --num_heads 1 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
            --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
            --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir2 --sparse_mode "strided" --sparse_context_window $cw

    done
done 


echo "=========================================="
echo "Sparse Attention Ablation Complete!"

