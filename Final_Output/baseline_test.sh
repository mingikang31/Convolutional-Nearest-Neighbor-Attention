#!/bin/bash 

### Conv-Test 

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
   output_dir1="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/Attention_NH3_s42"

   python main.py --layer Attention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
      --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
      --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir1

   output_dir2="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/Attention_NH1_s42"

   python main.py --layer Attention --patch_size 16 --num_layers 12 --num_heads 1 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
      --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
      --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir2


   # Local Attention Baseline 
   output_dir3="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/LocalAttention_NH3_s42"

   python main.py --layer LocalAttention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
      --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
      --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir3

   output_dir4="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/LocalAttention_NH1_s42"
   python main.py --layer LocalAttention --patch_size 16 --num_layers 12 --num_heads 1 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 \
      --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
      --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir4

      
   # # Neighborhood Attention
   # output_dir3="./Final_Output/Baseline_Test/ViT-Tiny-$(echo $dataset | awk '{print toupper($0)}')/NeighborhoodAttention_s42"
   # python main.py --layer NeighborhoodAttention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --K 3 \
   #     --dataset $dataset --resize 224 --batch_size 256 --num_epochs 150 --criterion CrossEntropy --optimizer adamw --weight_decay 1e-2 --lr $LR \
   #     --clip_grad_norm 1.0 --scheduler none --seed 42 --device cuda --output_dir $output_dir3

done
