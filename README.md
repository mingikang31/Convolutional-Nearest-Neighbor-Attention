# Convolutional Nearest Neighbor Attention (ConvNN-Attention) for Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Grants & Funding
- **Fall Research Grant**, Bowdoin College
- **Allen B. Tucker Computer Science Research Prize**, Bowdoin College
- **Christenfeld Summer Research Fellowship**, Bowdoin College
- **Google AI 2024 Funding**, Last Mile Fund
- **NYC Stem Funding**, Last Mile Fund

**Project periods:** Summer 2024, Spring 2025, Summer 2025, Fall 2026, Spring 2026

## Overview
Convolutional Nearest Neighbor Attention (ConvNN-Attention) is a novel attention mechanism featuring hard selection of nearest neighbors for Vision Transformers (ViTs). Traditional attention compute similarities across all token with soft selection, while ConvNN-Attention focuses on the k most relevant neighbors, enhancing efficiency and potentially improving performance.


### Key Concepts
- **Vision Transformer (ViT)**: A standard ViT architecture where the multi-head self-attention block can be replaced with our custom `MultiHeadConvNNAttention` and branching blocks.
- **Layer Flexibility**: The project supports single-block types (e.g., `ConvNNAttention` only) and branching architectures (e.g., `BranchingAttention`, `BranchingConv`) to combine the strengths of different operations.
- **Sampling Strategies**: To manage computational complexity, `ConvNN` layers support sampling strategies: `all` (dense), `random`, and `spatial`.

### Implementation

**ViT Layers (`vit_main.py`)**: These are 1D layers operating on sequences of patch embeddings.
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

## Training & Evaluation Examples

### Vision Transformer Examples

Train a standard ViT with default attention:

```shell
python main.py \
    --layer Attention \
    --patch_size 16 \
    --num_layers 8 \
    --num_heads 8 \
    --d_model 512 \
    --dataset cifar10 \
    --output_dir ./Output/ViT/Attention
```

Replace the attention block with our `MultiHeadConvNNAttention` layer:

```shell
python main.py \
    --layer ConvNNAttention \
    --patch_size 16 \
    --num_layers 8 \
    --num_heads 8 \
    --d_model 512 \
    --K 9 \
    --num_samples 4 \
    --sampling_type spatial \
    --dataset cifar10 \
    --output_dir ./Output/ViT/ConvNNAttention
```

## Command-Line Interface

### Vision Transformer (`main.py`)

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
| `--convolution_type`  | `standard`     | `standard`, `depthwise`, `depthwise-separable` | Convolution type for ConvNN-Attention layers.       |
| `--softmax_topk_val`  | `True`         | `True`, `False`                   | Use softmax on top-k values.           |
| `--K`                 | `9`            | *integer*                         | Number of nearest neighbors (k-NN) or kernel size.  |
| `--sampling_type`     | `all`          | `all`, `random`, `spatial`        | Sampling strategy for neighbor candidates.          |
| `--num_samples`       | `-1`           | *integer*                         | Number of samples for `random`/`spatial` modes. `-1` for all. |
| `--sample_padding`    | `0`            | *integer*                         | Padding for spatial sampling.                       |
| `--magnitude_type`    | `matmul`       | `cosine`, `euclidean`, `matmul`   | Similarity metric for nearest neighbors.            |
| `--coordinate_encoding` | `False`      | `True`, `False`                   | Enable coordinate encoding in ConvNN-Attention layers.        |
| `--branch_ratio`      | `0.5`          | *float (0-1)*                     | Branch ratio for branching ConvNN-Attention layers.           |

- (note) For `random` sampling type, set `--num_samples` to the desired number of neighbors to sample (e.g., `4`). For `spatial` sampling type, set `--num_samples` to the spatially separated grid (e.g., `3` for 3x3). For dense attention, set `--sampling_type` to `all` and `--num_samples` to `-1`.

#### Convolution Parameters (only for BranchConv block)

| Flag                  | Default        | Choices                           | Description                                         |
| --------------------- | -------------- | --------------------------------- | --------------------------------------------------- |
| `--kernel_size`       | `9`            | *integer*                         | Kernel size for Conv1d layers.                      |

#### Data Arguments

| Flag                  | Default        | Choices                           | Description                                         |
| --------------------- | -------------- | --------------------------------- | --------------------------------------------------- |
| `--dataset`           | `cifar10`      | `cifar10`, `cifar100`, `imagenet` | Dataset for training and evaluation.                |
| `--resize`            | `224`          | *integer*                         | Image resize dimension.                             |
| `--augment`           | `False`        | `True`, `False`                   | Enable data augmentation (random crop and flip).                           |
| `--noise`             | `0.0`          | *float*                           | Standard deviation of Gaussian noise to add.        |
| `--data_path`         | `./Data`       | *string*                          | Path to the dataset directory.                      |

#### Training Arguments

| Flag                  | Default        | Choices                                                    | Description                                    |
| --------------------- | -------------- | ---------------------------------------------------------- | ---------------------------------------------- |
| `--use_compiled`      | `False`        | `True`, `False`                                            | Use torch.compile for model optimization.      |
| `--compile_mode`      | `default`      | `default`, `reduce-overhead`, `reduce-memory`, `max-autotune` | Compilation mode for torch.compile.      |
| `--batch_size`        | `256`          | *integer*                                                  | Batch size for training and evaluation.        |
| `--num_epochs`        | `150`          | *integer*                                                  | Number of training epochs.                     |
| `--use_amp`           | `False`        | `True`, `False`                                            | Enable mixed precision training.               |
| `--clip_grad_norm`    | `1.0`          | *float*                                                    | Gradient clipping value.                       |

#### Loss & Optimizer Arguments

| Flag                  | Default        | Choices                           | Description                                         |
| --------------------- | -------------- | --------------------------------- | --------------------------------------------------- |
| `--criterion`         | `CrossEntropy` | `CrossEntropy`, `MSE`             | Loss function for training.                         |
| `--optimizer`         | `adamw`        | `adam`, `sgd`, `adamw`            | Optimizer for training.                             |
| `--momentum`          | `0.9`          | *float*                           | Momentum for SGD optimizer.                         |
| `--weight_decay`      | `0.05`         | *float*                           | Weight decay for Adam/AdamW optimizers.              |

#### Learning Rate Arguments

| Flag                  | Default        | Choices                           | Description                                         |
| --------------------- | -------------- | --------------------------------- | --------------------------------------------------- |
| `--lr`                | `0.0003`       | *float*                           | Learning rate for optimizer.                        |
| `--scheduler`         | `cosine`       | `step`, `cosine`, `plateau`, `none` | Learning rate scheduler type.                     |
| `--lr_step`           | `20`           | *integer*                         | Step size for StepLR scheduler.                     |
| `--lr_gamma`          | `0.1`          | *float*                           | Decay factor for StepLR scheduler.                  |

#### System & Output Arguments

| Flag                  | Default                    | Choices              | Description                                    |
| --------------------- | -------------------------- | -------------------- | ---------------------------------------------- |
| `--device`            | `cuda`                     | `cpu`, `cuda`, `mps` | Compute device for training/evaluation.        |
| `--seed`              | `0`                        | *integer*            | Random seed for reproducibility.               |
| `--output_dir`        | `./Output/VIT/`            | *string*             | Directory for output files and checkpoints.    |
| `--test_only`         | `False`                    | `True`, `False`      | Run model inference test only (no training).   |



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
