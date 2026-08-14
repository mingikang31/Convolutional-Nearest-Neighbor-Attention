cd /home/exouser/Convolutional-Nearest-Neighbor-Attention/

ulimit -s unlimited

# --- Safe Port Generation --- # 
export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
echo "Using Port: $MASTER_PORT"

# Supress warning
export OMP_NUM_THREADS=1

torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         vision_main.py \
        --model vit-base \
        --layer ConvNNAttention \
        --K 9 \
        --convolution_type standard \
        --sampling_type all \
        --num_samples -1 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_All_K9_StandardConv/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=9 Standard training failed"

torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         vision_main.py \
        --model vit-base \
        --layer ConvNNAttention \
        --K 9 \
        --convolution_type depthwise \
        --sampling_type all \
        --num_samples -1 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_All_K9_DepthwiseConv/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=9 Depthwise training failed"

torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         vision_main.py \
        --model vit-base \
        --layer ConvNNAttention \
        --K 9 \
        --convolution_type depthwise-separable \
        --sampling_type all \
        --num_samples -1 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH12_All_K9_DepthwiseSeparableConv/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=9 Depthwise Separable training failed"





echo "All training completed"
