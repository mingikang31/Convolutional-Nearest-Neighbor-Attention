cd /home/exouser/Convolutional-Nearest-Neighbor-Attention/

ulimit -s unlimited

# --- Safe Port Generation --- # 
export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
echo "Using Port: $MASTER_PORT"

# Supress warning
export OMP_NUM_THREADS=1

# torchrun --nproc_per_node=4 \
#          --rdzv_backend=c10d \
#          --rdzv_endpoint=localhost:$MASTER_PORT \
#          main.py \
#         --model vit-base \
#         --layer ConvNNAttention \
#         --K 9 \
#         --convolution_type depthwise \
#         --sampling_type all \
#         --num_samples -1 \
#         --num_heads 12 \
#         --dataset imagenet1k \
#         --data_path /home/exouser/Datasets \
#         --compile \
#         --use_amp \
#         --device cuda \
#         --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_All_K9/ \
#         --num_workers 16 \
#         --pin_memory \
#         --ddp \
#         --ddp_batch_size 320 || echo "K=9 all training failed"


# torchrun --nproc_per_node=4 \
#          --rdzv_backend=c10d \
#          --rdzv_endpoint=localhost:$MASTER_PORT \
#          main.py \
#         --model vit-base \
#         --layer ConvNNAttention \
#         --K 9 \
#         --convolution_type depthwise \
#         --sampling_type random \
#         --num_samples 32 \
#         --num_heads 12 \
#         --dataset imagenet1k \
#         --data_path /home/exouser/Datasets \
#         --compile \
#         --use_amp \
#         --device cuda \
#         --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_Rand_K9_N32/ \
#         --num_workers 16 \
#         --pin_memory \
#         --ddp \
#         --ddp_batch_size 320 || echo "K=9 random training failed"


# torchrun --nproc_per_node=4 \
#          --rdzv_backend=c10d \
#          --rdzv_endpoint=localhost:$MASTER_PORT \
#          main.py \
#         --model vit-base \
#         --layer ConvNNAttention \
#         --K 9 \
#         --convolution_type depthwise \
#         --sampling_type spatial \
#         --num_samples 128 \
#         --num_heads 12 \
#         --dataset imagenet1k \
#         --data_path /home/exouser/Datasets \
#         --compile \
#         --use_amp \
#         --device cuda \
#         --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_Spatial_K9_N128/ \
#         --num_workers 16 \
#         --pin_memory \
#         --ddp \
#         --ddp_batch_size 320 || echo "K=9 spatial training failed"

# torchrun --nproc_per_node=4 \
#          --rdzv_backend=c10d \
#          --rdzv_endpoint=localhost:$MASTER_PORT \
#          main.py \
#         --model vit-base \
#         --layer ConvNNAttention \
#         --K 16 \
#         --convolution_type depthwise \
#         --sampling_type all \
#         --num_samples -1 \
#         --num_heads 12 \
#         --dataset imagenet1k \
#         --data_path /home/exouser/Datasets \
#         --compile \
#         --use_amp \
#         --device cuda \
#         --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_All_K16/ \
#         --num_workers 16 \
#         --pin_memory \
#         --ddp \
#         --ddp_batch_size 320 || echo "K=16 all training failed"


torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         main.py \
        --model vit-base \
        --layer ConvNNAttention \
        --K 25 \
        --convolution_type depthwise \
        --sampling_type all \
        --num_samples -1 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_All_K25/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=25 all training failed"


torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         main.py \
        --model vit-base \
        --layer ConvNNAttention \
        --K 36 \
        --convolution_type depthwise \
        --sampling_type all \
        --num_samples -1 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_All_K36/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=36 all training failed"

echo "All training completed"
