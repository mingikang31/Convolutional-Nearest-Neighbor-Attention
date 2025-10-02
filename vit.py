"""VIT Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchsummary import summary 
import numpy as np


# from natten import NeighborhoodAttention1D, NeighborhoodAttention2D
from layers import (
    MultiHeadAttention, 
    MultiHeadConvNNAttention, 
    MultiHeadBranchingConv1d, 
    MultiHeadBranchingAttention,
    MultiHeadKvtAttention, 
    MultiHeadLocalAttention
)


'''VGG Model Class'''
class ViT(nn.Module): 
    def __init__(self, args): 
        super(ViT, self).__init__()
        assert args.img_size[1] % args.patch_size == 0 and args.img_size[2] % args.patch_size == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert args.d_hidden % args.num_heads == 0, "d_hidden must be divisible by n_heads"
        
        self.args = args
        self.args.model = "VIT"
        self.model = "VIT"
        
        self.d_hidden = self.args.d_hidden 
        self.d_mlp = self.args.d_mlp
        
        self.img_size = self.args.img_size[1:]
        self.n_classes = self.args.num_classes # Number of Classes
        self.n_heads = self.args.num_heads
        self.patch_size = (self.args.patch_size, self.args.patch_size) # Patch Size
        self.n_channels = self.args.img_size[0]
        self.n_layers = self.args.num_layers # Number of Layers
        
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        
        self.dropout = self.args.dropout # Dropout Rate
        self.attention_dropout = self.args.attention_dropout # Attention Dropout Rate   
        self.max_seq_length = self.n_patches + 1 # +1 for class token
        
        self.patch_embedding = PatchEmbedding(self.d_hidden, self.img_size, self.patch_size, self.n_channels) # Patch Embedding Layer
        self.positional_encoding = PositionalEncoding(self.d_hidden, self.max_seq_length)
        
        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(
            args=args, 
            d_hidden=self.d_hidden, 
            d_mlp=self.d_mlp, 
            num_heads=self.n_heads, 
            dropout=self.dropout, 
            attention_dropout=self.attention_dropout
            ) for _ in range(self.n_layers)])
        
        self.classifier = nn.Linear(self.d_hidden, self.n_classes)
        
        self.device = args.device
        
        self.to(self.device)
        self.name = f"{self.args.model} {self.args.layer}"
        
    def forward(self, x): 
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0]) # Taking the CLS token for classification
        return x

    def summary(self): 
        original_device = next(self.parameters()).device
        try:
            self.to("cpu")
            print(f"--- Summary for {self.name} ---")
            summary(self, input_size=self.img_size, device="cpu") 
        except Exception as e:
            print(f"Could not generate summary: {e}")
        finally:
            self.to(original_device)
        
    def parameter_count(self): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params
        
class PatchEmbedding(nn.Module): 
    def __init__(self, d_hidden, img_size, patch_size, n_channels=3): 
        super(PatchEmbedding, self).__init__()
        
        self.d_hidden = d_hidden # Dimensionality of Model 
        self.img_size = img_size # Size of Image
        self.patch_size = patch_size # Patch Size 
        self.n_channels = n_channels # Number of Channels in Image
        
        self.linear_projection = nn.Conv2d(in_channels=n_channels, out_channels=d_hidden, kernel_size=patch_size, stride=patch_size) # Linear Projection Layer
        self.norm = nn.LayerNorm(d_hidden) # Normalization Layer
        
        self.flatten = nn.Flatten(start_dim=2)
        
    def forward(self, x): 
        x = self.linear_projection(x) # (B, C, H, W) -> (B, d_hidden, H', W')
        x = self.flatten(x) # (B, d_hidden, H', W') -> (B, d_hidden, n_patches)
        x = x.transpose(1, 2) # (B, d_hidden, n_patches) -> (B, n_patches, d_hidden)
        x = self.norm(x) # (B, n_patches, d_hidden) -> (B, n_patches, d_hidden)
        return x
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_hidden, max_seq_length): 
        super(PositionalEncoding, self).__init__()
        
        self.cls_tokens = nn.Parameter(torch.randn(1, 1, d_hidden)) # Classification Token

        pe = torch.zeros(max_seq_length, d_hidden)  # Positional Encoding Tensor
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_hidden, 2).float() * (-np.log(10000.0) / d_hidden))  

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x): 
        # Expand to have class token for each image in batch 
        tokens_batch = self.cls_tokens.expand(x.shape[0], -1, -1) # (B, 1, d_hidden)
        
        # Concatenate class token with positional encoding
        x = torch.cat((tokens_batch, x), dim=1)
        
        # Add positional encoding to the input 
        x = x + self.pe[:, :x.size(1)].to(x.device) 
        return x

class TransformerEncoder(nn.Module): 
    def __init__(self, args, d_hidden, d_mlp, num_heads, dropout, attention_dropout):
        super(TransformerEncoder, self).__init__()
        self.args = args 

        self.d_hidden = d_hidden 
        self.d_mlp = d_mlp
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        convnn_attn_params = {
            "K": args.K, 
            "sampling_type": args.sampling_type,
            "num_samples": args.num_samples,
            "sample_padding": args.sample_padding,
            "magnitude_type": args.magnitude_type,
            "coordinate_encoding": args.coordinate_encoding
        }

        branching_conv_params = {
            "kernel_size": args.kernel_size,
            "K": args.K,
            "sampling_type": args.sampling_type,
            "num_samples": args.num_samples,
            "sample_padding": args.sample_padding,
            "magnitude_type": args.magnitude_type,
            "coordinate_encoding": args.coordinate_encoding,
            "branch_ratio": args.branch_ratio
        }

        branching_attn_params = {
            "K": args.K,
            "sampling_type": args.sampling_type,
            "num_samples": args.num_samples,
            "sample_padding": args.sample_padding,
            "magnitude_type": args.magnitude_type,
            "coordinate_encoding": args.coordinate_encoding,
            "branch_ratio": args.branch_ratio
        }

        # 1. Multi-Head Attention Layer
        if args.layer == "Attention":
            self.attention = MultiHeadAttention(d_hidden, num_heads, attention_dropout)

        # 2. ConvNN Attention Layer
        elif args.layer == "ConvNNAttention":
            self.attention = MultiHeadConvNNAttention(d_hidden, num_heads, attention_dropout, **convnn_attn_params)
            # self.attention = MultiHeadConvNNAttention_NoBatchSplit(d_hidden, num_heads, attention_dropout, **convnn_attn_params)

        # 3. Branching Conv1d Layer
        elif args.layer == "BranchConv":
            self.attention = MultiHeadBranchingConv1d(d_hidden, num_heads, attention_dropout, **branching_conv_params)
        # 4. Branching Attention Layer
        elif args.layer == "BranchAttention":
            self.attention = MultiHeadBranchingAttention(d_hidden, num_heads, attention_dropout, **branching_attn_params)
        # 5. Kvt Attention Layer
        elif args.layer == "KvtAttention":
            self.attention = MultiHeadKvtAttention(dim=d_hidden, num_heads=num_heads, attn_drop=attention_dropout, topk=args.K)

        # 6. Local Attention Layer
        elif args.layer == "LocalAttention":
            local_attention_params = {
                "window_size": 128,  # Default window size for local attention
                "dim_head": 64,  # Default dimension of each head
                "causal": False,  # Whether to use causal attention
                "prenorm": False,  # Whether to use pre-norm
                "qk_rmsnorm": False,  # Whether to use RMSNorm for query and key
                "qk_scale": 8,  # Scaling factor for query and key
                "use_xpos": False,  # Whether to use XPOS
                "xpos_scale_base": None,  # Base scale for XPOS
                "exact_windowsize": None,  # Exact window size for local attention
                "gate_values_per_head": False  # Whether to gate values per head
                
            }
            self.attention = MultiHeadLocalAttention(dim=d_hidden, heads=num_heads, dropout=attention_dropout, **local_attention_params)

        # 7. Neighborhood Attention Layer
        elif args.layer == "NeighborhoodAttention": 
            neighborhood_attention_params = {
                "stride": 1,  # Default stride for neighborhood attention
                "dilation": 1,  # Default dilation for neighborhood attention
                "qkv_bias": True,  # Whether to use bias in QKV projections
                "qk_scale": None,  # Scaling factor for QK
                "is_causal": False,  # Whether to use causal attention
            }
            
            self.attention = NeighborhoodAttention1D(embed_dim=d_hidden, num_heads=num_heads, kernel_size=args.K, proj_drop=attention_dropout, **neighborhood_attention_params
            ) 

        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Multilayer Perceptron 
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_mlp),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_hidden)
        )
        
    def forward(self, x): 
        # Pre-Norm Multi-Head Attention 
        norm_x = self.norm1(x) 
        attn_output = self.attention(norm_x)  
        x = x + self.dropout1(attn_output)
        
        # Post-Norm Feed Forward Network
        norm_x = self.norm2(x)  
        mlp_output = self.mlp(norm_x)
        x = x + self.dropout2(mlp_output)  
        return x