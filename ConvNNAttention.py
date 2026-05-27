# Torch + Numpy
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

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
            self.conv[0].weight.data.fill_(1.0)


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

"""Sampled ConvNN Attention Implementation (Random & Spatial Sampling)"""
class MultiHeadConvNNAttention_Sampled(nn.Module):
    def __init__(self, 
                 d_hidden, 
                 num_heads, 
                 attention_dropout, 
                 K, 
                 num_samples, 
                 sampling_type='random', 
                 sample_padding=0, 
                 convolution_type='depthwise', 
                 seq_length=197
                 ):

        super(MultiHeadConvNNAttention_Sampled, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
    
        self.d_hidden = d_hidden 
        self.num_heads = num_heads 
        self.attention_dropout = attention_dropout 
        self.d_k = d_hidden // num_heads 
        self.seq_length = seq_length 
        
        self.K = K 
        self.num_samples = num_samples 
        self.sampling_type = sampling_type
        self.sample_padding = sample_padding 

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

    def _get_sample_indices(self, seq_length, device):
        if self.sampling_type == 'random': 
            return torch.randperm(seq_length, device=device)[:self.num_samples]
        elif self.sampling_type == 'spatial':
            return torch.linspace(
                0 + self.sample_padding, 
                seq_length - self.sample_padding - 1, 
                self.num_samples, 
                device=device
            ).long()

    def _prime_N(self, v, qk, K, sample_idx):
        # v: (B*num_heads, d_k, seq_length), qk: (B*num_heads, num_samples, seq_length)
        b, c, t = v.shape 
        topk_value, topk_indices = torch.topk(qk, k=K-1, dim=2, largest=True)

        # Map sample indices back to original matrix positions 
        mapped_tensor = sample_idx[topk_indices]
        token_indices = torch.arange(t, device=v.device).view(1, t, 1).expand(b, t, 1)
        final_indices = torch.cat([token_indices, mapped_tensor], dim=-1)
        topk_indices_exp = final_indices.unsqueeze(1).expand(b, c, t, K)

        # Expand topk values to match the shape of indices       
        topk_values_exp = topk_value.unsqueeze(1).expand(b, c, t, K-1)
        ones = torch.ones((b, c, t, 1), device=v.device)
        topk_values_exp = torch.cat((ones, topk_values_exp), dim=-1)

        # softmax
        topk_values_exp = torch.softmax(topk_values_exp, dim=-1)

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

        sample_idx = self._get_sample_indices(x.shape[1], x.device)
        k_sample = k[:, :, sample_idx, :]
        attn_matrix = torch.matmul(q, k_sample.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Mask 
        range_idx = torch.arange(len(sample_idx), device=x.device)
        attn_matrix[:, :, sample_idx, range_idx] = float('-inf')

        v_merged = v.reshape(B * self.num_heads, self.seq_length, self.d_k).permute(0, 2, 1)
        am_merged = attn_matrix.reshape(B * self.num_heads, self.seq_length, self.num_samples)

        # Prime and Convolution 
        prime = self._prime_N(v_merged, am_merged, self.K, sample_idx)
        out = self.conv(prime) # (B*num_heads, d_k, seq_length
        out = out.permute(0, 2, 1).contiguous().view(B, self.num_heads, self.seq_length, self.d_k)
        out = self.dropout(out)
        output = self.W_o(self.combine_heads(out)) # (B, SL, d_hidden)
        return output
