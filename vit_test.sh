#!/bin/bash

# ViT-Tiny Configuration:
# num_layers: 12
# d_hidden: 192
# d_mlp: 768
# num_heads: 3

# ### Baseline Models ###

# # Attention
# python vit_main.py --layer Attention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/ViT-Tiny/Attention

# # KVT Attention
# python vit_main.py --layer KvtAttention --patch_size 16 --num_layers 12 --K 9 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny/KvtAttention

# # ConvNN All
# python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny/ConvNN_All

# # ConvNN Random
# python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/TEST/VIT-Tiny/ConvNN_Random

# # ConvNN Spatial
# python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/ViT-Tiny/ConvNN_Spatial

# # ConvNNAttention All
# python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 12 --K 9 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/ViT-Tiny/ConvNNAttention_All

# # ConvNNAttention Random
# python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/ViT-Tiny/ConvNNAttention_Random

# # ConvNNAttention Spatial
# python vit_main.py --layer ConvNNAttention --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/ViT-Tiny/ConvNNAttention_Spatial

# # Local Attention
# python vit_main.py --layer LocalAttention --patch_size 16 --num_layers 12 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/ViT-Tiny/LocalAttention


# # NeighborhoodAttention
# python vit_main.py --layer NeighborhoodAttention --patch_size 16 --num_layers 12 --K 9 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/ViT-Tiny/NeighborhoodAttention



### N Test Experiments ###
# - ConvNN Random Samples but varying N
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 8 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Random_N8

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 16 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Random_N16

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Random_N32

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 64 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Random_N64

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type random --num_samples 128 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Random_N128

# - ConvNN Spatial Samples but varying N
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 8 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Spatial_N8

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 16 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Spatial_N16

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 32 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Spatial_N32

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 64 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Spatial_N64

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type spatial --num_samples 128 --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-N-Test/ConvNN_Spatial_N128

### K Test Experiments ###
# - ConvNN All Samples but varying K
python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 1 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K1

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 3 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K3

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 5 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K5

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 7 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K7

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 9 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K9

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 11 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K11

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 13 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K13

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 16 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K16

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 25 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K25

python vit_main.py --layer ConvNN --patch_size 16 --num_layers 12 --K 36 --sampling_type all --num_heads 3 --d_hidden 192 --d_mlp 768 --dropout 0.1 --attention_dropout 0.1 --dataset cifar10 --num_epochs 10 --seed 0 --output_dir ./Output/VIT-Tiny-K-Test/ConvNN_All_K36

echo "All experiments finished."


