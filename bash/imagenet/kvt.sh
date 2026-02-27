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
         main.py \
        --model vit-base \
        --layer KvtAttention \
        --K 2 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-KvtAttention_NH12_K2/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=2 training failed"


torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         main.py \
        --model vit-base \
        --layer KvtAttention \
        --K 4 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-KvtAttention_NH12_K4/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=4 training failed"


torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         main.py \
        --model vit-base \
        --layer KvtAttention \
        --K 9 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-KvtAttention_NH12_K9/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=9 training failed"

torchrun --nproc_per_node=4 \
         --rdzv_backend=c10d \
         --rdzv_endpoint=localhost:$MASTER_PORT \
         main.py \
        --model vit-base \
        --layer KvtAttention \
        --K 16 \
        --num_heads 12 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-KvtAttention_NH12_K16/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320 || echo "K=16 training failed"

echo "All training completed"
