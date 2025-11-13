"""layers.py - Multi-Head Layers for Transformer Encoder"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch import einsum
import numpy as np 
from typing import Optional

from utils import default, LocalAttention, l2norm, exists, rearrange, apply_rotary_pos_emb

"""(*) PixelShuffle1D"""
class PixelShuffle1D(nn.Module): 
    """
    1D Pixel Shuffle Layer for Convolutional Neural Networks.
    
    Attributes: 
        upscale_factor (int): Upscale factor for pixel shuffle. 
        
    Notes:
        Input's channel size must be divisible by the upscale factor. 
    """
    
    def __init__(self, upscale_factor):
        """ 
        Initializes the PixelShuffle1D module.
        
        Parameters:
            upscale_factor (int): Upscale factor for pixel shuffle.
        """
        super(PixelShuffle1D, self).__init__()
        
        self.upscale_factor = upscale_factor

    def forward(self, x): 
        batch_size, channel_len, token_len = x.shape[0], x.shape[1], x.shape[2]
        
        output_channel_len = channel_len / self.upscale_factor 
        if output_channel_len.is_integer() == False: 
            raise ValueError('Input channel length must be divisible by upscale factor')
        output_channel_len = int(output_channel_len)
        
        output_token_len = int(token_len * self.upscale_factor)
        
        x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
        
        return x 

"""(*) PixelUnshuffle1D"""
class PixelUnshuffle1D(nn.Module):  
    """
    1D Pixel Unshuffle Layer for Convolutional Neural Networks.
    
    Attributes:
        downscale_factor (int): Downscale factor for pixel unshuffle.
        
    Note:
        Input's token size must be divisible by the downscale factor
    
    """
    
    def __init__(self, downscale_factor):
        """
        Intializes the PixelUnshuffle1D module.
        
        Parameters:
            downscale_factor (int): Downscale factor for pixel unshuffle.
        """
        super(PixelUnshuffle1D, self).__init__()
        
        self.downscale_factor = downscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        channel_len = x.shape[1]
        token_len = x.shape[2]

        output_channel_len = int(channel_len * self.downscale_factor)
        output_token_len = token_len / self.downscale_factor
        
        if output_token_len.is_integer() == False:
            raise ValueError('Input token length must be divisible by downscale factor')
        output_token_len = int(output_token_len)
        
        x = torch.reshape(x, (batch_size, output_channel_len, output_token_len)).contiguous()
        
        return x 
    
"""Multi-Head Layers for Transformer Encoder"""
class MultiHeadAttention(nn.Module): 
    def __init__(self, d_hidden, num_heads, attention_dropout):
        super(MultiHeadAttention, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads # dimension of each head
        self.dropout = nn.Dropout(attention_dropout)
        
        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)        
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = self.dropout(torch.softmax(attn_scores, dim=-1))
        output = torch.matmul(attn_probs, V)
        return output, attn_probs
    
    def split_head(self, x): 
        batch_size, seq_length, d_hidden = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden) 
    
    def forward(self, x, mask=None):
        q = self.split_head(self.W_q(x)) # (B, num_heads, seq_length, d_k)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))
        
        attn_output, _ = self.scaled_dot_product_attention(q, k, v, mask) # (B, num_heads, seq_length, d_k)
        output = self.W_o(self.combine_heads(attn_output)) # (B, seq_length, d_hidden)
        return output

class MultiHeadConvNNAttention(nn.Module):
    def __init__(self, 
                 d_hidden, 
                 num_heads, 
                 attention_dropout,
                 K, 
                 sampling_type, 
                 num_samples, 
                 sample_padding, 
                 magnitude_type, 
                 seq_length=197, 
                 coordinate_encoding=False, 
                 convolution_type='depthwise', 
                 softmax_topk_val=True
                 ):
        
        super(MultiHeadConvNNAttention, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        # Core Parameters
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.d_k = d_hidden // num_heads

        # ConvNN Parameters
        self.K = K
        self.seq_length = seq_length

        # 3 types of sampling: all, random, spatial
        self.sampling_type = sampling_type
        self.num_samples = int(num_samples) 
        self.sample_padding = int(sample_padding) if sampling_type == 'spatial' else 0

        # Similarity Metric 
        self.magnitude_type = magnitude_type
        self.maximum = True if self.magnitude_type in ('cosine', 'matmul') else False

        # Coordinate Encoding (optional) 
        self.coordinate_encoding = coordinate_encoding
        self.coordinate_cache = {}
        
        # Linear projections for query, key, value
        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)   
        self.dropout = nn.Dropout(attention_dropout)

        self.in_channels = (d_hidden // num_heads) + 1 if coordinate_encoding else (d_hidden // num_heads)
        self.out_channels = (d_hidden // num_heads) 

        # Convolution Layer 
        self.convolution_type = convolution_type

        if convolution_type == 'standard':
            self.conv = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.K,
                stride=self.K,
                padding=0,
                bias=False
            )
        elif convolution_type == 'depthwise':
            self.conv = nn.Conv1d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=self.K,
                stride=self.K,
                padding=0,
                groups=self.in_channels, 
                bias=False
            )
            self.conv.weight.data.fill_(1.0)
        elif convolution_type == 'depthwise-separable':
            self.conv = nn.Sequential(
                # Depthwise Convolution
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=self.K,
                    stride=self.K,
                    padding=0,
                    groups=self.in_channels,
                    bias=False
                ), 
                # Pointwise Convolution
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0, 
                    bias=False
                )
            )

        # Softmax 
        self.softmax_topk_val = softmax_topk_val
        
        # Utility Variables 
        self.INF = 1.1
        self.NEG_INF = -0.1 
        
    def split_head(self, x):
        batch_size, seq_length, d_hidden = x.size()
        self.batch_size = batch_size
        return x.contiguous().view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def batch_combine(self, x):
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)

    def batch_split(self, x):
        if self.num_heads == 1:
            return x.unsqueeze(1)
        else:
            x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
            return x.permute(0, 1, 3, 2).contiguous()

    def combine_heads(self, x):
        if self.num_heads == 1:
            return x.squeeze(1) 
        else:
            batch_size, _, seq_length, d_k = x.size()
            return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden)
        
    def forward(self, x):
        # Note: x shape: (B, seq_length, d_hidden)
        # 1. Splithead & Batch Combine
        k = self.batch_combine(self.split_head(self.W_k(x)))
        v = self.batch_combine(self.split_head(self.W_v(x)))
        

        # 2. Add Coordinate Encoding 
        k = self._add_coordinate_encoding(k) if self.coordinate_encoding else k
        v = self._add_coordinate_encoding(v) if self.coordinate_encoding else v

        # 3. Sampling & Similarity Calculation
        if self.sampling_type == 'all': # All Samples
            q = self.batch_combine(self.split_head(self.W_q(x)))
            
            q = self._add_coordinate_encoding(q) if self.coordinate_encoding else q

            similarity_matrix = self._calculate_matmul_matrix(k, q) if self.magnitude_type == 'matmul' else self._calculate_cosine_matrix(k, q) if self.magnitude_type == 'cosine' else self._calculate_euclidean_matrix(k, q, sqrt=True)

            
            prime = self._prime(v, similarity_matrix, self.K, self.maximum) if not self.softmax_topk_val else self._prime_softmax(v, similarity_matrix, self.K, self.maximum)

        elif self.sampling_type == 'random': # Random Samples
            rand_idx = torch.randperm(x.shape[1], device=x.device)[:self.num_samples]
            x_sample = x[:, rand_idx, :]            
            q = self.batch_combine(self.split_head(self.W_q(x_sample)))
            q = self._add_coordinate_encoding(q) if self.coordinate_encoding else q

            similarity_matrix = self._calculate_matmul_matrix(k, q) if self.magnitude_type == 'matmul' else self._calculate_cosine_matrix(k, q) if self.magnitude_type == 'cosine' else self._calculate_euclidean_matrix(k, q, sqrt=True)

            range_idx = torch.arange(len(rand_idx), device=q.device)
            similarity_matrix[:, rand_idx, range_idx] = self.INF if self.magnitude_type == 'euclidean' else self.NEG_INF

            prime = self._prime_N(v, similarity_matrix, self.K, rand_idx, self.maximum) if not self.softmax_topk_val else self._prime_softmax_N(v, similarity_matrix, self.K, rand_idx, self.maximum)

        elif self.sampling_type == 'spatial': # Spatial Samples
            spat_idx = torch.linspace(0 + self.sample_padding, x.shape[1] - self.sample_padding - 1, self.num_samples, device=x.device).long()
            x_sample = x[:, spat_idx, :]
            q = self.batch_combine(self.split_head(self.W_q(x_sample)))
            q = self._add_coordinate_encoding(q) if self.coordinate_encoding else q

            similarity_matrix = self._calculate_matmul_matrix(k, q) if self.magnitude_type == 'matmul' else self._calculate_cosine_matrix(k, q) if self.magnitude_type == 'cosine' else self._calculate_euclidean_matrix(k, q, sqrt=True)
            
            range_idx = torch.arange(len(spat_idx), device=q.device)
            similarity_matrix[:, spat_idx, range_idx] = self.INF if self.magnitude_type == 'euclidean' else self.NEG_INF

            prime = self._prime_N(v, similarity_matrix, self.K, spat_idx, self.maximum) if not self.softmax_topk_val else self._prime_softmax_N(v, similarity_matrix, self.K, spat_idx, self.maximum)
            
        else: 
            raise ValueError("Invalid sampling_type. Must be one of ['all', 'random', 'spatial']")

        # 4. Conv1d Layer
        x = self.conv(prime)  

        # 5. Dropout + Reshape (B, seq_length, d_hidden)
        x = self.dropout(x)
        x = x.permute(0, 2, 1) 

        # 6. Final Linear Projection
        x = self.W_o(self.combine_heads(self.batch_split(x)))
        return x       

    def _calculate_matmul_matrix(self, K, Q):
        attn_matrix = torch.matmul(K.transpose(1, 2), Q) / self.d_k ** 0.5
        return attn_matrix 
        
    def _calculate_euclidean_matrix(self, K, Q, sqrt=False):
        k_norm_squared = torch.sum(K**2, dim=1, keepdim=True)
        q_norm_squared = torch.sum(Q**2, dim=1, keepdim=True)
        dot_product = torch.bmm(K.transpose(1, 2), Q)

        dist_matrix = k_norm_squared.transpose(1, 2) + q_norm_squared - 2 * dot_product
        dist_matrix = torch.sqrt(dist_matrix) if sqrt else dist_matrix
        return dist_matrix 

    def _calculate_cosine_matrix(self, K, Q):
        k_norm = F.normalize(K, p=2, dim=1)
        q_norm = F.normalize(Q, p=2, dim=1)
        similarity_matrix = torch.matmul(k_norm.transpose(1, 2), q_norm)
        return similarity_matrix

    def _prime(self, v, qk, K, maximum):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K, dim=2, largest=maximum)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)

        # Normalize by K for distance metrics 
        if not maximum: 
            prime = prime / (topk_values_exp + 1e-8)
        else:
            prime = topk_values_exp * prime 

        prime = prime.view(b, c, -1)

        return prime

    def _prime_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K-1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map sample indicies back to original matrix positions 
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=-1)
        topk_indices_exp = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand topk values to match the shape of indices
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K-1)
        ones = torch.ones((b, c, t, 1), device=v.device)
        topk_values_exp = torch.cat((ones, topk_values_exp), dim=-1)

        # Gather matrix values and apply similarity weighting 
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()    
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        
        if not maximum:  # euclidean distance
            prime = prime / (topk_values_exp + 1e-8)
        else:
            prime = topk_values_exp * prime
        prime = prime.view(b, c, -1)
        return prime

    def _prime_softmax(self, v, qk, K, maximum):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K, dim=2, largest=maximum)
        
        # Apply softmax
        if maximum:  # cosine/matmul (maximize)
            topk_values = torch.softmax(topk_values, dim=-1)
        else:  # euclidean (minimize) - negate before softmax
            topk_values = torch.softmax(-topk_values, dim=-1)

        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime
        prime = prime.view(b, c, -1)
        return prime

    def _prime_softmax_N(self, v, qk, K, rand_idx, maximum):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K-1, dim=2, largest=maximum)
        tk = topk_indices.shape[-1]
        assert K == tk + 1, "Error: K must be same as tk + 1. K == tk + 1."

        # Map sample indices back to original matrix positions 
        mapped_tensor = rand_idx[topk_indices]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=-1)
        topk_indices_exp = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand topk values to match the shape of indices
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K-1)
        ones = torch.ones((b, c, t, 1), device=v.device)
        topk_values_exp = torch.cat((ones, topk_values_exp), dim=-1)
        
        # Apply softmax
        if maximum:
            topk_values_exp = torch.softmax(topk_values_exp, dim=-1)
        else:
            topk_values_exp = torch.softmax(-topk_values_exp, dim=-1)
                
        # Gather matrix values and apply similarity weighting 
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()    
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime

        prime = prime.view(b, c, -1)
        return prime

    def _add_coordinate_encoding(self, x):
        b, c, t = x.shape 
        cache_key = f"{b}_{t}_{x.device}"
        if cache_key in self.coordinate_cache: 
            expanded_coords = self.coordinate_cache[cache_key]
        else: 
            coords_vec = torch.linspace(start=-1, end=1, steps=t, device=x.device).unsqueeze(0).expand(b, -1) 
            expanded_coords = coords_vec.unsqueeze(1).expand(b, -1, -1) 
            self.coordinate_cache[cache_key] = expanded_coords

        x_with_coords = torch.cat([x, expanded_coords], dim=1) 
        return x_with_coords 

class MultiHeadConv1dAttention(nn.Module):
    def __init__(self, d_hidden, num_heads, kernel_size): 
        super(MultiHeadConv1dAttention, self).__init__()
    
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.d_k = d_hidden // num_heads
        
        self.kernel_size = kernel_size
        self.stride = 1
        
        self.W_x = nn.Linear(d_hidden, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_hidden)

        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads
        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding="same"
        )
        
    def split_head(self, x): 
        batch_size, seq_length, d_hidden = x.size()
        self.batch_size = batch_size
        self.seq_length = seq_length
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden) 
    
    def batch_split(self, x): 
        x = x.reshape(self.batch_size, -1, self.d_k, self.seq_length)
        return x.permute(0, 1, 3, 2).contiguous()
        
    def batch_combine(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        x = x.permute(0, 1, 3, 2).contiguous() 
        return x.view(-1, self.d_k, seq_length)       
    
    def forward(self, x):
        x = self.batch_combine(self.split_head(self.W_x(x)))
        x = self.conv(x) 
        x = self.W_o(self.combine_heads(self.batch_split(x.permute(0, 2, 1))))
        return x

class MultiHeadBranchingConv(nn.Module):
    def __init__(self,  
                 d_hidden, 
                 num_heads, 
                 attention_dropout,
                 kernel_size, 
                 K, 
                 sampling_type, 
                 num_samples, 
                 sample_padding, 
                 magnitude_type, 
                 seq_length=197, 
                 coordinate_encoding=False, 
                 convolution_type='depthwise',
                 softmax_topk_val=True,
                 branch_ratio=0.5
                 ):
        super(MultiHeadBranchingConv, self).__init__()

        # Attention Parameters
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        # Conv1d Parameters 
        self.kernel_size = kernel_size

        # ConvNN Parameters
        self.K = K
        self.sampling_type = sampling_type
        self.num_samples = int(num_samples)
        self.sample_padding = int(sample_padding) if sampling_type == 'spatial' else 0
        self.magnitude_type = magnitude_type
        self.seq_length = seq_length
        self.coordinate_encoding = coordinate_encoding

        self.branch_ratio = branch_ratio

        self.d_hidden_convnn = int(self.branch_ratio * d_hidden)
        self.d_hidden_conv1d = d_hidden - self.d_hidden_convnn

        if self.branch_ratio != 0: 
            self.convnn = MultiHeadConvNNAttention(
                d_hidden=self.d_hidden_convnn, 
                num_heads=num_heads, 
                attention_dropout=attention_dropout,
                K=K, 
                sampling_type=sampling_type, 
                num_samples=num_samples, 
                sample_padding=sample_padding, 
                magnitude_type=magnitude_type, 
                seq_length=seq_length, 
                coordinate_encoding=coordinate_encoding, 
                convolution_type=convolution_type,
                softmax_topk_val=softmax_topk_val
            )
        if self.branch_ratio != 1:
            self.conv1d = MultiHeadConv1dAttention(
                d_hidden=self.d_hidden_conv1d, 
                num_heads=num_heads, 
                kernel_size=kernel_size
            )

        self.pointwise_linear = nn.Linear(d_hidden, d_hidden)
        
        
    def forward(self, x):
        if self.branch_ratio == 0:
            return self.conv1d(x)
        elif self.branch_ratio == 1:
            return self.convnn(x)
        else:
            x1 = self.convnn(x[:, :, :self.d_hidden_convnn])
            x2 = self.conv1d(x[:, :, self.d_hidden_convnn:])
            out = torch.cat((x1, x2), dim=2)
            out = self.pointwise_linear(out)
            return out

class MultiHeadBranchingAttention(nn.Module):
    def __init__(self,  
                 d_hidden, 
                 num_heads, 
                 attention_dropout,
                 K, 
                 sampling_type, 
                 num_samples, 
                 sample_padding, 
                 magnitude_type, 
                 seq_length=197, 
                 coordinate_encoding=False, 
                 convolution_type='depthwise',
                 softmax_topk_val=True,
                 branch_ratio=0.5
                 ):
        super(MultiHeadBranchingAttention, self).__init__()

        # Attention Parameters
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        # ConvNN Parameters 
        self.K = K
        self.sampling_type = sampling_type 
        self.num_samples = int(num_samples)
        self.sample_padding = int(sample_padding) if sampling_type == 'spatial' else 0
        self.magnitude_type = magnitude_type
        self.seq_length = seq_length
        self.coordinate_encoding = coordinate_encoding

        self.branch_ratio = branch_ratio

        self.d_hidden_convnn = int(self.branch_ratio * d_hidden)
        self.d_hidden_attention = d_hidden - self.d_hidden_convnn

        if self.branch_ratio != 0:
            self.convnn = MultiHeadConvNNAttention(
                d_hidden=self.d_hidden_convnn, 
                num_heads=num_heads, 
                attention_dropout=attention_dropout,
                K=K, 
                sampling_type=sampling_type, 
                num_samples=num_samples, 
                sample_padding=sample_padding, 
                magnitude_type=magnitude_type, 
                seq_length=seq_length, 
                coordinate_encoding=coordinate_encoding,
                convolution_type=convolution_type,
                softmax_topk_val=softmax_topk_val
            )

        if self.branch_ratio != 1:
            self.attention = MultiHeadAttention(
                d_hidden=self.d_hidden_attention, 
                num_heads=num_heads, 
                attention_dropout=attention_dropout
            )

        self.pointwise_linear = nn.Linear(d_hidden, d_hidden)

    def forward(self, x):
        if self.branch_ratio == 0:
            return self.convnn(x)
        elif self.branch_ratio == 1:
            return self.attention(x)
        else:
            x1 = self.convnn(x[:, :, :self.d_hidden_convnn])
            x2 = self.attention(x[:, :, self.d_hidden_convnn:])
            out = torch.cat((x1, x2), dim=2)
            out = self.pointwise_linear(out)
            return out
        
class MultiHeadKvtAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,topk=100):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)
        self.topk = topk

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # the core code block
        mask=torch.zeros(B,self.num_heads,N,N,device=x.device,requires_grad=False)
        index=torch.topk(attn,k=self.topk,dim=-1,largest=True)[1]
        mask.scatter_(-1,index,1.)
        attn=torch.where(mask>0,attn,torch.full_like(attn,float('-inf')))
        # end of the core code block

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadLocalAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size = 128,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None,
        exact_windowsize = None,
        gate_values_per_head = False,
    ):
        super().__init__()        
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim) if prenorm else None

        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.qk_rmsnorm = qk_rmsnorm

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.causal = causal
        self.window_size = window_size
        self.exact_windowsize = default(exact_windowsize, True)

        self.attn_fn = LocalAttention(
            dim = dim_head,
            window_size = window_size,
            causal = causal,
            dropout = dropout, 
            autopad = True,
            scale = (qk_scale if qk_rmsnorm else None),
            exact_windowsize = self.exact_windowsize,
            use_xpos = use_xpos,
            xpos_scale_base = xpos_scale_base,
        )

        self.to_v_gate = None

        if gate_values_per_head:
            self.to_v_gate = nn.Sequential(
                nn.Linear(dim, heads)
            )

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        attn_bias = None,
        cache = None,
        return_cache = False
    ):
        seq_len = x.shape[-2]

        if exists(self.norm):
            x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            q = q * self.q_scale
            k = k * self.k_scale

        if exists(cache):
            assert seq_len == 1

            assert self.causal and not exists(mask), 'only allow caching for specific configuration'

            ck, cv = cache

            q = q * (q.shape[-1] ** -0.5)

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            effective_window_size = self.attn_fn.look_backward * self.window_size

            if self.exact_windowsize:
                kv_start_index = -(effective_window_size + 1)
            else:
                seq_len = k.shape[-2]
                kv_start_index = -(effective_window_size + (seq_len % self.window_size))

            k, v = tuple(t[..., kv_start_index:, :] for t in (k, v))

            if exists(self.attn_fn.rel_pos):
                rel_pos = self.attn_fn.rel_pos
                pos_emb, xpos_scale = rel_pos(k)
                q, k = apply_rotary_pos_emb(q, k, pos_emb, scale = xpos_scale)

            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            if exists(attn_bias):
                k_len = k.shape[-2]
                attn_bias = attn_bias[..., -1:, -k_len:]
                assert attn_bias.shape[-1] == sim.shape[-1]
                sim = sim + attn_bias

            attn = sim.softmax(dim = -1)
            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        else:
            out = self.attn_fn(q, k, v, mask = mask, attn_bias = attn_bias)

        if return_cache:
            kv = torch.stack((k, v))

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            gates = rearrange(gates, 'b n h -> b h n 1')
            out = out * gates.sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if not return_cache:
            return out

        return out, kv



### Code from : https://github.com/openai/sparse_attention/blob/master/utils.py ###
def get_attn_mask(n, attn_mode, local_attn_ctx=None, device='cuda'):
    if attn_mode == 'all':
        # ✓ BIDIRECTIONAL - all patches attend to all patches
        b = torch.ones(n, n, device=device)
    
    elif attn_mode == 'local':
        # ✓ BIDIRECTIONAL LOCAL - attend to nearby patches in both directions
        bandwidth = local_attn_ctx
        # Create a band matrix (not just lower triangular)
        b = torch.zeros(n, n, device=device)
        for i in range(n):
            start = max(0, i - bandwidth // 2)
            end = min(n, i + bandwidth // 2 + 1)
            b[i, start:end] = 1
    
    elif attn_mode == 'strided':
        # ✓ BIDIRECTIONAL STRIDED
        stride = local_attn_ctx
        x = torch.arange(n, dtype=torch.int32, device=device).view(n, 1)
        y = x.t()
        q = x.expand(n, n)
        k = y.expand(n, n)
        # Remove c1 = q >= k (this was the causal constraint!)
        c2 = ((q - k).abs() % stride) == 0  # Distance is multiple of stride
        b = c2.float()
    
    b = b.view(1, 1, n, n)
    return b


def strided_transpose(x, n_ctx, local_attn_ctx, blocksize=None):
    """
    Transpose for strided attention pattern.
    
    Args:
        x: tensor of shape [batch, seq_len, embd]
        n_ctx: context length
        local_attn_ctx: stride length
        blocksize: not used in PyTorch version (kept for API compatibility)
    
    Returns:
        transposed tensor
    """
    bT_ctx = n_ctx // local_attn_ctx
    n, t, embd = x.shape
    x = x.view(n, bT_ctx, local_attn_ctx, embd)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(n, t, embd)
    return x


def split_heads(x, n_heads):
    """
    Split the last dimension into (n_heads, depth).
    Transpose to shape [batch, n_heads, seq_len, depth]
    """
    batch_size, seq_len, d_model = x.shape
    depth = d_model // n_heads
    x = x.view(batch_size, seq_len, n_heads, depth)
    return x.permute(0, 2, 1, 3)


def merge_heads(x):
    """
    Merge heads back to original shape.
    Input: [batch, n_heads, seq_len, depth]
    Output: [batch, seq_len, d_model]
    """
    batch_size, n_heads, seq_len, depth = x.shape
    x = x.permute(0, 2, 1, 3)
    return x.reshape(batch_size, seq_len, n_heads * depth)


def attention_impl(q, k, v, n_heads, attention_dropout, attn_mode, local_attn_ctx=None):
    """
    Standard attention implementation with different masking patterns.
    
    Args:
        q, k, v: query, key, value tensors of shape [batch, seq_len, d_model]
        n_heads: number of attention heads
        attn_mode: attention pattern ('all', 'local', 'strided')
        local_attn_ctx: context window for local/strided attention
    
    Returns:
        attention output of shape [batch, seq_len, d_model]
    """
    # Split heads: [batch, n_heads, seq_len, depth]
    q = split_heads(q, n_heads)
    k = split_heads(k, n_heads)
    v = split_heads(v, n_heads)
    
    # Get attention mask
    n_timesteps = k.shape[2]
    mask = get_attn_mask(n_timesteps, attn_mode, local_attn_ctx, device=q.device)
    
    # Scaled dot-product attention
    # [batch, n_heads, seq_len, seq_len]
    depth = q.shape[-1]
    scale_amount = 1.0 / np.sqrt(depth)
    
    # Compute attention scores
    w = torch.matmul(q, k.transpose(-2, -1))
    w = w * scale_amount
    
    # Apply mask (using large negative value for masked positions)
    w = w * mask + -1e9 * (1 - mask)
    
    # Softmax
    w = F.softmax(w, dim=-1)

    w = F.dropout(w, p=attention_dropout)
    
    # Apply attention to values
    a = torch.matmul(w, v)
    
    # Merge heads
    a = merge_heads(a)
    
    return a


class MultiHeadSparseAttention(nn.Module):
    """
    Multi-head sparse attention module.
    
    Supports different attention patterns: 'all', 'local', 'strided'
    """
    def __init__(self, d_hidden, num_heads, attention_dropout, attn_mode='all', local_attn_ctx=None):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.attn_mode = attn_mode
        self.local_attn_ctx = local_attn_ctx
        
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.k_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.v_proj = nn.Linear(d_hidden, d_hidden, bias=False)
        self.out_proj = nn.Linear(d_hidden, d_hidden, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: input tensor of shape [batch, seq_len, d_model]
        
        Returns:
            output tensor of shape [batch, seq_len, d_model]
        """
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Apply attention
        attn_output = attention_impl(
            q, k, v, 
            self.num_heads, 
            self.attention_dropout,
            self.attn_mode, 
            self.local_attn_ctx
        )
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


# For gradient checkpointing (equivalent to @recomputable decorator)
def checkpoint_attention(q, k, v, n_heads, attn_mode, local_attn_ctx=None):
    """
    Attention with gradient checkpointing to save memory.
    """
    return torch.utils.checkpoint.checkpoint(
        attention_impl,
        q, k, v, n_heads, attn_mode, local_attn_ctx,
        use_reentrant=False
    )


def strided_attention_impl(q, k, v, n_heads, local_attn_ctx, blocksize=32):
    """
    Strided attention with transposition (as in blocksparse version).
    
    Note: This is the dense implementation. For true block-sparse computation,
    you would need a custom CUDA kernel or library like Triton.
    """
    n_ctx = q.shape[1]
    
    # Apply strided transpose
    q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
    k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
    v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    
    # Apply attention
    a = attention_impl(q, k, v, n_heads, 'strided', local_attn_ctx)
    
    # Reverse the transpose
    n, t, embd = a.shape
    bT_ctx = n_ctx // local_attn_ctx
    a = a.view(n, local_attn_ctx, bT_ctx, embd)
    a = a.permute(0, 2, 1, 3)
    a = a.reshape(n, t, embd)
    
    return a
