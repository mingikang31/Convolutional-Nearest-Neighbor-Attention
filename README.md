# Convolutional Nearest Neighbor Attention (ConvNN-Attention) for Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Paper 
- **Attention Via Convolutional Nearest Neighbors**, Mingi Kang, Jeova Farias, [[arXiv](https://arxiv.org/abs/2511.14137)] 
- This repository is complemented by the Convolutional Nearest Neighbor (ConvNN) repository: [https://github.com/mingikang31/Convolutional-Nearest-Neighbor](https://github.com/mingikang31/Convolutional-Nearest-Neighbor)

## Grants & Funding
- **Fall Research Grant**, Bowdoin College
- **Allen B. Tucker Computer Science Research Prize**, Bowdoin College
- **Christenfeld Summer Research Fellowship**, Bowdoin College
- **Google AI 2024 Funding**, Last Mile Fund
- **NYC Stem Funding**, Last Mile Fund

**Project periods:** Summer 2024, Spring 2025, Summer 2025, Fall 2026, Spring 2026

## Overview
Convolutional Nearest Neighbor Attention (ConvNN-Attention) is a novel attention mechanism featuring hard selection of nearest neighbors for Vision Transformers (ViTs). Traditional attention compute similarities across all token with soft selection, while ConvNN-Attention focuses on the $k$ most relevant neighbors with convolutional operations to aggregate information, enhancing efficiency and potentially improving performance.

### Key Concepts
- **Vision Transformer (ViT)**: A standard ViT architecture where the multi-head self-attention block can be replaced with our custom `MultiHeadConvNNAttention` and branching blocks.
- **Layer Flexibility**: The project supports single-block types (e.g., `MultiHeadConvNNAttention` only) and branching architectures (e.g., `BranchingAttention`, `BranchingConv`) to combine the strengths of different operations.
- **Sampling Strategies**: To manage computational complexity, `ConvNN` layers support sampling strategies: `all` (dense), `random`, and `spatial`.
- **Number of Heads**: `MultiHeadConvNNAttention` does not use head splitting. Each block processes the full feature dimension.

### Implementation

**ViT Layers (`vit.py`)**: ViT implementation with modular attention layers.

**Attention Layers (`layers.py`)**: Implements various attention mechanisms:
- **`MultiHeadAttention`**: Standard Transformer multi-head self-attention
- **`MultiHeadConvNNAttention`**: Convolutional Nearest Neighbor Attention layer with k-NN selection
- **`MultiHeadKvtAttention`**: An implementation of **k-NN Attention for boosting Vision Transformers**
- **`MultiHeadLocalAttention`**: Local Attention implementation from **lucidrains**
- **`NeighborhoodAttention1D`**: NeighborhoodAttention1D from **NATTEN**
- **`MultiHeadBranchingConv`**: Branching attention layer combining convolution and ConvNN-Attention.
- **`MultiHeadBranchingAttention`**: Branching attention layer combining standard attention and ConvNN-Attention.


## Installation
```shell
git clone [https://github.com/mingikang31/Convolutional-Nearest-Neighbor-Attention.git](https://github.com/mingikang31/Convolutional-Nearest-Neighbor-Attention.git)
cd Convolutional-Nearest-Neighbor-Attention
```

Then, install the required dependencies:

```shell
pip install -r requirements.txt
```

## Command-Line Interface

### Main Script (`main.py`)

Run `python main.py --help` to see all available options.

#### Model & Layer Configuration

| Flag                  | Default     | Choices                                                                    | Description                                     |
| --------------------- | ----------- | -------------------------------------------------------------------------- | ----------------------------------------------- |
| `--layer`             | `Attention` | `Attention`, `ConvNNAttention`, `KvtAttention`, `LocalAttention`, `NeighborhoodAttention`, `BranchConv`, `BranchAttention` | Layer type for ViT transformer blocks.          |
| `--patch_size`        | `16`        | *integer*                                                                  | Patch size for attention models.                |
| `--num_layers`        | `12`        | *integer*                                                                  | Number of transformer encoder layers.           |
| `--num_heads`         | `3`         | *integer*                                                                  | Number of attention heads.                      |
| `--d_hidden`          | `192`       | *integer*                                                                  | Hidden dimension for the model.                 |
| `--d_mlp`             | `768`       | *integer*                                                                  | MLP dimension for the model.                    |
| `--dropout`           | `0.1`       | *float*                                                                    | Dropout rate for linear layers.                 |
| `--attention_dropout` | `0.1`       | *float*                                                                    | Dropout rate for attention layers.              |

#### ConvNN-Attention Specific Parameters

| Flag                  | Default        | Choices                           | Description                                         |
| --------------------- | -------------- | --------------------------------- | --------------------------------------------------- |
| `--convolution_type`  | `depthwise`     | `standard`, `depthwise`, `depthwise-separable` | Convolution type for ConvNN-Attention layers.       |
| `--softmax_topk_val`  | `True`         | `True`, `False`                   | Use softmax on top-k values.           |
| `--K`                 | `9`            | *integer*                         | Number of nearest neighbors (k-NN) or kernel size.  |
| `--sampling_type`     | `all`          | `all`, `random`, `spatial`        | Sampling strategy for neighbor candidates.          |
| `--num_samples`       | `-1`           | *integer*                         | Number of samples for `random`/`spatial` modes. `-1` for all. |
| `--sample_padding`    | `0`            | *integer*                         | Padding for spatial sampling.                       |
| `--magnitude_type`    | `matmul`       | `cosine`, `euclidean`, `matmul`   | Similarity metric for nearest neighbors.            |
| `--coordinate_encoding` | `False`      | `True`, `False`                   | Enable coordinate encoding in ConvNN-Attention layers.        |

- (note) For `random` sampling type, set `--num_samples` to the desired number of neighbors to sample (e.g., `4`). For `spatial` sampling type, set `--num_samples` to the spatially separated grid (e.g., `3` for 3x3). For dense attention, set `--sampling_type` to `all` and `--num_samples` to `-1`.

#### Convolution-Specific Parameters

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--kernel_size` | `9` | *integer* | kernel size for convolution 1D |

#### Branching Layer Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--branch_ratio` | `0.5` | Branch ratio for `Branching` layer (between 0 and 1). Example: `0.25` means 25% of input/output channels go to ConvNN-Attention branch, rest to Attention branch |

#### Dataset Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--dataset` | `cifar10` | `cifar10`, `cifar100`, `imagenet` | Dataset to use for training and evaluation |
| `--data_path` | `./Data` | *path* | Path to the dataset directory |
| `--resize` | `224` | *integer* | Resize images to specified size (e.g., 224 for 224x224) |
| `--augment` | `False` | *flag* | Enable data augmentation |
| `--noise` | `0.0` | *float* | Standard deviation of Gaussian noise to add to data |

#### Training Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--batch_size` | `256` | *integer* | Batch size for training and evaluation |
| `--num_epochs` | `150` | *integer* | Number of epochs for training |
| `--use_amp` | `False` | *flag* | Enable mixed precision training (automatic mixed precision) |
| `--use_compiled` | `False` | *flag* | Use `torch.compile` for model optimization |
| `--compile_mode` | `default` | `default`, `reduce-overhead`, `reduce-memory`, `max-autotune` | Compilation mode for `torch.compile` |
| `--clip_grad_norm` | `1.0` | *float* | Gradient clipping maximum norm value |

#### Optimization Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--criterion` | `CrossEntropy` | `CrossEntropy`, `MSE` | Loss function to use for training |
| `--optimizer` | `adamw` | `adam`, `sgd`, `adamw` | Optimizer algorithm |
| `--lr` | `1e-3` | *float* | Initial learning rate |
| `--weight_decay` | `1e-6` | *float* | Weight decay (L2 regularization) for optimizer |
| `--momentum` | `0.9` | *float* | Momentum parameter for SGD optimizer |
| `--scheduler` | `none` | `step`, `cosine`, `plateau`, `none` | Learning rate scheduler type |
| `--lr_step` | `20` | *integer* | Step size for step scheduler (decrease LR every N epochs) |
| `--lr_gamma` | `0.1` | *float* | Multiplicative factor for LR decay in step scheduler |

#### System Configuration

| Flag | Default | Choices | Description |
|------|---------|---------|-------------|
| `--device` | `cuda` | `cpu`, `cuda`, `mps` | Device to use for training and evaluation |
| `--seed` | `0` | *integer* | Random seed for reproducibility |

#### Output Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--output_dir` | `./Output/VGG/ConvNN` | Directory to save output files (results, model info, logs) |
| `--test_only` | `False` | *flag* | Only test the model without training |


## Training Examples

### Train ViT-Tiny with Standard self-Attention

```shell
python main.py \
    --layer Attention \
    --num_layers 12 \
    --num_heads 3 \
    --d_hidden 192 \
    --d_mlp 768 \
    --dataset cifar10 \
    --optimizer adamw \
    --lr 1e-3 \
    --num_epochs 150 \
    --output_dir ./Output/ViT/Attention
```

### Train ViT-Tiny with Pure ConvNN

```shell
python main.py \
    --layer ConvNNAttention \
    --convolution_type depthwise \
    --num_layers 12 \
    --d_hidden 192 \
    --d_mlp 768 \
    --K 9 \
    --sampling_type spatial \
    --num_samples 8 \
    --similarity_type Col \
    --aggregation_type Col \
    --dataset cifar10 \
    --optimizer adamw \
    --lr 1e-3 \
    --num_epochs 150 \
    --output_dir ./Output/ViT/ConvNNAttention
```

### Test Mode Only

```shell
python main.py \
    --layer Attention \
    --test_only \
    --device cuda
```


## Output Files
After running training, the following files are saved in `--output_dir`:

- **`args.txt`**: All command-line arguments used for the experiment
- **`model.txt`**: Model architecture and parameter summary
- **`train_eval_results.txt`**: Training and evaluation results (loss, accuracy per epoch)

## Supported Architectures & Layers

### Models
- **ViT**: Vision Transformer with modular attention layers

### Layers
- **`MultiHeadAttention`**: Standard multi-head self-attention
- **`MultiHeadConvNNAttention`**: Convolutional Nearest Neighbor Attention
- **`MultiHeadBranchingConv`**: Branching layer combining Conv2d and ConvNN-Attention
- **`MultiHeadBranchingAttention`**: Branching layer combining standard attention and ConvNN-Attention
- **`MultiHeadKvtAttention`**: k-NN Attention for Vision Transformers
- **`MultiHeadLocalAttention`**: Local Attention from lucidrains
- **`NeighborhoodAttention1D`**: Neighborhood Attention from NATTEN

## Notes

- Set `--num_samples -1` with `--sampling_type all` to use all spatial locations as candidates
- `--branch_ratio 0.0` means 100% Conv2d (baseline)
- `--branch_ratio 1.0` means 100% ConvNN
- `--branch_ratio 0.5` means 50% Conv2d and 50% ConvNN
- Use `--use_compiled` for significant speedups on long training runs (first epoch slower due to compilation)
- Mixed precision (`--use_amp`) can reduce memory usage by ~50% with minimal accuracy impact



## Project Structure

```
.
├── layers.py            # Attention modules for ViT
├── dataset.py           # CIFAR-10/100 & ImageNet wrappers
├── train_eval.py        # Training & evaluation loop
├── vit.py               # Vision Transformer implementation
├── main.py              # CLI entrypoint for ViT Experiments
├── utils.py             # I/O, logging, seed setup
├── requirements.txt     # Python dependencies
├── README.md            # ← you are here
├── Data/                # Datasets (CIFAR-10/100, ImageNet)
├── Output/              # Training outputs and checkpoints
└── LICENSE              # MIT License
```

## License

Convolutional-Nearest-Neighbor is released under the MIT License. Please see the [LICENSE](https://www.google.com/search?q=LICENSE) file for more information.

## Contributing

Contributions, issues, and feature requests are welcome\!
Please reach out to:

  - **Mingi Kang** [mkang2@bowdoin.edu](mailto:mkang2@bowdoin.edu)
  - **Jeova Farias** [j.farias@bowdoin.edu](mailto:j.farias@bowdoin.edu)

<!-- end list -->
