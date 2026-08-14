import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from layers import MultiHeadAttention

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

"""ConvNN Attention implementation with For-Loop for number of heads (Sanity Check)"""
class MultiHeadConvNNAttention_ForLoop(nn.Module):
    def __init__(self, 
                 d_hidden, 
                 num_heads, 
                 attention_dropout,
                 K, 
                 seq_length=197, 
                 ):
        
        super(MultiHeadConvNNAttention_ForLoop, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        # Core Parameters
        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.d_k = d_hidden // num_heads

        # ConvNN Parameters
        self.K = K
        self.seq_length = seq_length
        # Linear projections for query, key, value
        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)   

        self.W_q.weight.data.fill_(2.0)
        self.W_k.weight.data.fill_(3.0)
        self.W_v.weight.data.fill_(4.0)
        self.W_o.weight.data.fill_(5.0)

        self.dropout = nn.Dropout(attention_dropout)

        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads
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
        
    def split_head(self, x): 
        batch_size, seq_length, d_hidden = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)
        
    def combine_heads(self, x): 
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden) 

    def process_heads(self, v, attn_matrix):
        B, NH, SL, DK = v.shape

        outs = []

        for i in range(NH):
            v_i = v[:, i, :, :]
            am_i = attn_matrix[:, i, :, :]
            print("v_i shape: ", v_i.shape)
            print("am_i shape: ", am_i.shape)

            prime_i = self._prime(v_i.transpose(1, 2), am_i, self.K)
            print("prime_i shape: ", prime_i.shape)

            out_i = self.conv(prime_i).transpose(1, 2).unsqueeze(1)
            print("out_i shape: ", out_i.shape)
            
            outs.append(out_i)


        out = torch.concat(outs, dim=1)
        out = self.dropout(out)
        return out 

    def _prime(self, v, qk, K):
        b, c, t = v.shape
        topk_values, topk_indices = torch.topk(qk, k=K, dim=2, largest=True)
        
        # Apply softmax
        topk_values = torch.softmax(topk_values, dim=-1)

        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)

        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime
        prime = prime.view(b, c, -1)
        return prime
        
    
    def forward(self, x): 
        q = self.split_head(self.W_q(x)) # (B, num_heads, seq_length, d_k)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))

        print("q|k|v shape: ", q.shape) 

        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        print("attn scores shape: ", attn_matrix.shape) 
        print() 

        attn_output = self.process_heads(v, attn_matrix)
        print()
        print("attn output shape: ", attn_output.shape)
        
        output = self.W_o(self.combine_heads(attn_output)) # (B, seq_length, d_hidden)
        return output

"""[NOT IN USE] Old ConvNN Attention Implementation"""
class MultiHeadConvNNAttention_Old(nn.Module):
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
        
        super(MultiHeadConvNNAttention_Old, self).__init__()
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

"""[NOT IN USE]"""
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

"""[NOT IN USE]"""
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

"""[NOT IN USE]"""
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















