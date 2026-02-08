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
        --layer ConvNNAttention \
        --K 9 \
        --convolution_type depthwise \
        --softmax_topk_val \
        --sampling_type random \
        --num_samples 32 \
        --num_heads 1 \
        --dataset imagenet1k \
        --data_path /home/exouser/Datasets \
        --compile \
        --use_amp \
        --device cuda \
        --output_dir /home/exouser/Convolutional-Nearest-Neighbor-Attention/Output/ImageNet1K/ViT-Base-ConvNNAttention_NH1_Rand_K9_N32/ \
        --num_workers 16 \
        --pin_memory \
        --ddp \
        --ddp_batch_size 320


## [MHA Attention NH=12] ##
### ViT-Base Training Log on JetStream2 H100 g5.4xl ###
# 4x NVIDIA H100 80GB HBM3 GPUs

### gpustat log during training ###
# Batch Size: 320 per GPU, Total Batch Size: 1280
# imagenet1k                Thu Feb  5 19:48:33 2026  580.126.09
# [0] NVIDIA H100 80GB HBM3 | 68°C,  99 % | 56072 / 81559 MB | exouser(56030M)
# [1] NVIDIA H100 80GB HBM3 | 71°C, 100 % | 56072 / 81559 MB | exouser(56030M)
# [2] NVIDIA H100 80GB HBM3 | 71°C,  99 % | 56072 / 81559 MB | exouser(56030M)
# [3] NVIDIA H100 80GB HBM3 | 70°C, 100 % | 56222 / 81559 MB | exouser(56180M)

## Training Log from Current Configuration ##
# Total time = 180 s. * 300 epochs ~ 14.5 Hours
# [Epoch 001] Time: 288.7117s | [Train] Loss: 7.00203794 Accuracy: Top1: 0.0000%, Top5: 0.0000% | [Test] Loss: 6.87562466 Accuracy: Top1: 0.2946%, Top5: 1.1889%
# [Epoch 002] Time: 178.1445s | [Train] Loss: 6.75060904 Accuracy: Top1: 0.0000%, Top5: 0.0000% | [Test] Loss: 5.85781240 Accuracy: Top1: 4.9354%, Top5: 13.8390%


## [ConvNN-Attention NH=1 Rand K=9 N=32] ##
### ViT-Base Training Log on JetStream2 H100 g5.4xl ###
# 4x NVIDIA H100 80GB HBM3 GPUs

### gpustat log during training ###
# Batch Size: 320 per GPU, Total Batch Size: 1280
# imagenet1k                Fri Feb  6 17:30:58 2026  580.126.09
# [0] NVIDIA H100 80GB HBM3 | 59°C, 100 % | 72936 / 81559 MB | exouser(72908M)
# [1] NVIDIA H100 80GB HBM3 | 60°C, 100 % | 72980 / 81559 MB | exouser(72952M)
# [2] NVIDIA H100 80GB HBM3 | 61°C, 100 % | 72936 / 81559 MB | exouser(72908M)
# [3] NVIDIA H100 80GB HBM3 | 61°C,  97 % | 72980 / 81559 MB | exouser(72952M)

## Training Log from Current Configuration ##
# Total time = 590 s. * 300 epochs ~ 49.2 Hours
# [Epoch 001] Time: 699.0366s | [Train] Loss: 7.00485574 Accuracy: Top1: 0.0000%, Top5: 0.0000% | [Test] Loss: 6.88593769 Accuracy: Top1: 0.2969%, Top5: 1.1562%
# [Epoch 002] Time: 609.2266s | [Train] Loss: 6.79427098 Accuracy: Top1: 0.0000%, Top5: 0.0000% | [Test] Loss: 6.11738300 Accuracy: Top1: 3.1016%, Top5: 9.6094%