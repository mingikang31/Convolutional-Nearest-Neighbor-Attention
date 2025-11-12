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

# Main Loop 
for dataset in "${DATASETS[@]}"; do

   # Attention Baseline
   output_dir1="./Final_Output/Baseline_Test_Sparse/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/SparseAttention_All_NH1_s42"

   python main.py --layer SparseAttention --patch_size 16 --num_layers 12 --num_heads 1 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
      --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
      --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir1 --sparse_mode "all" --sparse_block_size 32 --sparse_context_window 128



    output_dir2="./Final_Output/Baseline_Test_Sparse/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/SparseAttention_Local_NH1_s42"

    python main.py --layer SparseAttention --patch_size 16 --num_layers 12 --num_heads 1 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
       --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
       --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir2 --sparse_mode "local" --sparse_block_size 32 --sparse_context_window 128

   output_dir3="./Final_Output/Baseline_Test_Sparse/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/SparseAttention_Strided_NH1_s42"

   python main.py --layer SparseAttention --patch_size 16 --num_layers 12 --num_heads 1 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
      --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
      --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir3 --sparse_mode "strided" --sparse_block_size 32 --sparse_context_window 128


done

echo "=========================================="
echo "Sparse Attention Complete!"