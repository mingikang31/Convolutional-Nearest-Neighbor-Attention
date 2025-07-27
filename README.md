# Convolutional Nearest Neighbor Attention (ConvNNAttention) with Vision Transformer Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Grants & Funding
- **Christenfeld Summer Research Fellowship**, Bowdoin College
- **Google AI 2024 Funding**, Last Mile Fund
- **NYC Stem Funding**, Last Mile Fund

**Project periods:** Summer 2024, Spring 2025, Summer 2025

## Overview
Traditional convolutions aggregate features from a fixed grid of adjacent pixels. This project introduces **Convolutional Nearest Neighbor (ConvNN)**, a novel layer that operates on a more flexible, data-driven neighborhood.

**How ConvNN Works:**
- For each token (a pixel or feature vector), it finds the *K*-nearest neighbors from a global or sampled pool of candidates.
- It aggregates the values of these neighbors, often using distance or similarity metrics for weighting.
- This approach allows the model to capture long-range dependencies and non-local relationships that standard convolutions might miss.
- ConvNN is designed to be a drop-in replacement for standard layers in both fully convolutional networks and Vision Transformers.

### Key Concepts
- **Vision Transformer (ViT)**: A standard ViT architecture where the multi-head self-attention block can be replaced with our custom `MultiHeadConvNN` or `MultiHeadConvNNAttention` layers.
- **Layer Flexibility**: The project supports single-layer types (e.g., `ConvNN` only) and powerful branching architectures (e.g., `Conv2d/ConvNN`, `Attention/ConvNN_Attn`) to combine the strengths of different operations.
- **Sampling Strategies**: To manage computational complexity, `ConvNN` layers support multiple sampling strategies: `all` (dense), `random`, and `spatial`.

### Implementation

**ViT Layers (`vit_main.py`)**: These are 1D layers operating on sequences of patch embeddings.
- **`MultiHeadAttention`**: Standard Transformer multi-head self-attention.
- **`MultiHeadConvNN`**: A 1D ConvNN using linear projections on sequences.
- **`MultiHeadConvNNAttention`**: A 1D ConvNN using linear projections on features (Q,K,V).
- **`MultiHeadConv1d` / `MultiHeadConv1dAttention`**: Traditional 1D convolutions integrated into the attention block structure.
- **`MultiHeadKvtAttention`**: An implementation of k-NN Attention for boosting Vision Transformers.
- **`MultiHeadLocalAttention`**: Local Attention implementation from lucidrains
- **`NeighborhoodAttention1D`**: NeighborhoodAttention1D from NATTEN

## Installation
```shell
git clone [https://github.com/mingikang31/Convolutional-Nearest-Neighbor-Attention.git](https://github.com/mingikang31/Convolutional-Nearest-Neighbor-Attention.git)
cd Convolutional-Nearest-Neighbor-Attention
````

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
    --num_samples 32 \
    --sampling_type random \
    --dataset cifar10 \
    --output_dir ./Output/ViT/ConvNNAttention
```

## Command-Line Interface

### AllConvNet (`allconvnet_main.py`)

Run `python allconvnet_main.py --help` to see all available options.

#### Model & Layer Configuration

| Flag                | Default                 | Choices                                                                                               | Description                                         |
| ------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `--layer`           | `ConvNN`                | `Conv2d`, `ConvNN`, `ConvNN_Attn`, `Attention`, `Conv2d/ConvNN`, `Conv2d/Attention`, etc.             | Which convolution/attention/branching layer to use. |
| `--num_layers`      | `5`                     | *integer*                                                                                             | Number of sequential layers in the network.         |
| `--channels`        | `8 16 32 64 128`        | *string of integers with space between*                                                               | Output channel sizes for each respective layer.     |
| `--kernel_size`     | `3`                     | *integer*                                                                                             | Kernel size for standard `Conv2d` layers.           |
| `--num_heads`       | `4`                     | *integer*                                                                                             | Number of heads for `Attention2d` layers.           |
| `--shuffle_pattern` | `BA`                    | `BA`, `NA`                                                                                            | `unshuffle` Before & `shuffle` After, or `None`.    |
| `--shuffle_scale`   | `2`                     | *integer*                                                                                             | Scale factor for pixel shuffle/unshuffle.           |

#### ConvNN-Specific Parameters (for AllConvNet)

| Flag               | Default        | Choices                    | Description                                         |
| ------------------ | -------------- | -------------------------- | --------------------------------------------------- |
| `--K`              | `9`            | *integer*                  | Number of nearest neighbors to find.                |
| `--sampling_type`  | `all`          | `all`, `random`, `spatial` | How to select the pool of neighbor candidates.      |
| `--num_samples`    | `-1`           | *integer*                  | Number of samples for `random` or `spatial` mode. Set to -1 for `all` sampling. |
| `--sample_padding` | `0`            | *integer*                  | Padding for `spatial` sampling.                     |
| `--magnitude_type` | `similarity`   | `similarity`, `distance`   | Metric for finding nearest neighbors.               |
| `--coordinate_encoding` | `true`    | `flag to use`              | Add `Coordinate Encoding layers to the data`        |


### Vision Transformer (`vit_main.py`)

Run `python main.py --help` to see all available options.

#### Model & Layer Configuration

| Flag                  | Default     | Choices                                                                    | Description                                     |
| --------------------- | ----------- | -------------------------------------------------------------------------- | ----------------------------------------------- |
| `--layer`             | `Attention` | `Attention`, `ConvNN`, `ConvNNAttention`, `Conv1d`, `Conv1dAttention`, `KvtAttention`, `LocalAttention`, `NeighborhoodAttention` | Layer type for ViT transformer blocks.          |
| `--patch_size`        | `16`        | *integer*                                                                  | Side length of square image patches.            |
| `--num_layers`        | `8`         | *integer*                                                                  | Number of transformer encoder layers.           |
| `--num_heads`         | `8`         | *integer*                                                                  | Number of attention heads.                      |
| `--d_model`           | `512`       | *integer*                                                                  | The main embedding dimension of the model.      |
| `--d_mlp`             | `2048`      | *integer*                                                                  | Dimension of the inner MLP feed-forward layer.  |
| `--dropout`           | `0.1`       | *float*                                                                    | General dropout rate for linear layers.         |
| `--attention_dropout` | `0.1`       | *float*                                                                    | Dropout rate applied to attention probabilities. |

*(Note: To change other parameters for `LocalAttention` and `NeighborhoodAttention`, change them in `TransformerEncoder` class in vit.py`.)*
#### ConvNN-Specific Parameters (for ViT)

| Flag               | Default        | Choices                    | Description                                         |
| ------------------ | -------------- | -------------------------- | --------------------------------------------------- |
| `--K`              | `9`            | *integer*                  | Number of nearest neighbors to find or kernel size. |
| `--sampling_type`  | `all`          | `all`, `random`, `spatial` | How to select the pool of neighbor candidates.      |
| `--num_samples`    | `-1`           | *integer*                  | Number of samples for `random` or `spatial` mode. Set to -1 for `all` sampling. |
| `--sample_padding` | `0`            | *integer*                  | Padding for `spatial` sampling.                     |
| `--magnitude_type` | `similarity`   | `similarity`, `distance`   | Metric for finding nearest neighbors.               |

*(Note: Data, Training, and Optimization arguments are shared between both scripts. Run `--help` for details.)*

## Project Structure

```
.
├── layers1d.py          # 1D layers for ViT and AllConvNet
├── layers2d.py          # 2D layers for AllConvNet
├── dataset.py           # CIFAR-10/100 & ImageNet wrappers
├── train_eval.py        # Training & evaluation loop
├── allconvnet.py        # AllConvNet architecture implementation
├── allconvnet_main.py   # CLI entrypoint for AllConvNet
├── vit.py               # Vision Transformer implementation
├── vit_main.py          # CLI entrypoint for ViT
├── utils.py             # I/O, logging, seed setup
├── README.md            # ← you are here
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
