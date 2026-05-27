"""
FlashConvNNAttention.py

- Implementation of ConvNNAttention with Depthwise Convolution using CUDA/C++ for fused operations and pointer manipulation.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

'''STILL WORKING ON CUDA IMPLEMENTATION WITH 1x1 K KERNELS USING POINTERS INSTEAD OF GATHERING NEAREST NEIGHBORS INTO A LARGE PRIME TENSOR'''
class FlashConvNNAttention(nn.Module):
    # TODO finish implementation
    def __init__(self, 
                 d_hidden, 
                 num_heads, 
                 attention_dropout, 
                 K, 
                 convolution_type = 'depthwise', 
                 seq_length=197):
        super(FlashConvNNAttention, self).__init__()

        self.d_hidden = d_hidden 
        self.num_heads = num_heads 
        self.attention_dropout = attention_dropout 
        self.d_k = d_hidden // num_heads 
        self.K = K 
        self.seq_length = seq_length

        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)

        self.dropout = nn.Dropout(attention_dropout)

        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads

        self.k_convs = [] 
        
        if convolution_type == 'standard':
            for _ in range(K):
                conv = nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                )
                conv.weight.data.fill_(1.0)
                self.k_convs.append(conv)
        elif convolution_type == 'depthwise':
            for _ in range(K):
                conv = nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.in_channels,
                    bias=False
                )
                conv.weight.data.fill_(1.0)
                self.k_convs.append(conv)
                
    def split_head(self, x):
        batch_size, seq_length, d_hidden = x.size() 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size() 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden)
    
    def forward(self, x):
        return 0 
    
        # B = x.shape[0]
        
        # q = self.split_head(self.W_q(x)) # (B, NH, SL, DK)
        # k = self.split_head(self.W_k(x)) # (B, NH, SL, DK)
        # v = self.split_head(self.W_v(x)) # (B, NH, SL, DK)

        # attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k) # (B, NH, SL, SL)

        # v_merged = v.reshape(B * self.num_heads, self.seq_length, self.d_k).permute(0, 2, 1) # (B*NH, DK, SL)
        # am_merged = attn_matrix.reshape(B * self.num_heads, self.seq_length, self.seq_length) # (B*NH, SL, SL)

        # v_primes = [] 
        # for i in range(self.K):
        #     v_out_i = self.k_convs[i](v_merged) # (B*NH, DK, SL)
        #     v_primes.append(v_out_i)

        # for i in range(self.K):
        #     topk_values, topk_indices = torch.topk(am_merged, k=self.K, dim=2, largest=True)
        #     topk_values = torch.softmax(topk_values, dim=-1)

        #     #



"""Regular ConvNN Attention Implementation"""
class MultiHeadConvNNAttention(nn.Module):
    def __init__(self, 
                 d_hidden,
                 num_heads, 
                 attention_dropout, 
                 K, 
                 convolution_type='depthwise',
                 seq_length=197):

        super(MultiHeadConvNNAttention, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        self.d_hidden = d_hidden 
        self.num_heads = num_heads 
        self.attention_dropout = attention_dropout 
        self.d_k = d_hidden // num_heads 
        self.K = K 
        self.seq_length = seq_length

        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)

        self.dropout = nn.Dropout(attention_dropout)

        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads

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
        self.conv.weight.data.fill_(1.0)

    def split_head(self, x):
        batch_size, seq_length, d_hidden = x.size() 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size() 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden)

    def _prime(self, v, qk, K):
        # v: (B*num_heads, d_k, seq_length), qk: (B*num_heads, seq_length, seq_length)
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K, dim=2, largest=True)
        print("Top-K Values Shape:", topk_values.shape) # (B*num_heads, seq_length, K)
        print("Top-K Indices Shape:", topk_indices.shape) # (B*num_heads, seq_length, K)
        print(topk_indices[0, 0, :5]) # Print first 5 indices for the first head of the first batch
        print(v[0, 0, :5]) # Print first 5 values for the first head of the first batch

        topk_values = torch.softmax(topk_values, dim=-1)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime
        prime = prime.view(b, c, -1)
        return prime

    def forward(self, x):
        B = x.shape[0]

        # Linear Projection + Split Heads 
        q = self.split_head(self.W_q(x)) # (B, NH, SL, DK)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))

        # Attention Matrix: (B, NH, SL, SL) - Q @ K^T
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Merge B and num_heads into dim for prime & conv 
        ## (B, NH, SL, DK) → (B*NH, DK, SL) for v and (B, NH, SL, SL) → (B*NH, SL, SL)
        v_merged = v.reshape(B * self.num_heads, self.seq_length, self.d_k).permute(0, 2, 1)
        am_merged = attn_matrix.reshape(B * self.num_heads, self.seq_length, self.seq_length)

        # Prime and Convolution 
        prime = self._prime(v_merged, am_merged, self.K)
        out = self.conv(prime) # (B*num_heads, d_k, seq_length) 

        # Reshape back: (B*NH, DK, SL) → (B, NH, SL, DK)
        out = out.permute(0, 2, 1).contiguous().view(B, self.num_heads, self.seq_length, self.d_k)
        out = self.dropout(out) 

        # Combine Heads and Final Linear Projection
        output = self.W_o(self.combine_heads(out)) # (B, SL, d_hidden)
        return output

conv = MultiHeadConvNNAttention(d_hidden=24, num_heads=1, attention_dropout=0.0, K=4, convolution_type='depthwise')
ex = torch.randn(1, 197, 24)
out = conv(ex)