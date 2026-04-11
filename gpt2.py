"""
GPT2 Model implementation in PyTorch. 

Original Code from OpenAI's GPT-2 repository, written in TensorFlow. 

https://github.com/openai/gpt-2/blob/master/src/model.py
"""

import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd

class GPT2(nn.Module):
    def __init__(self, 
                 args, 
                 vocab_size=50257, 
                 max_seq_length=1024,
                 embedding_dim=768,
                 num_attention_heads=12,
                 num_layers=12,
                 dropout=0.1,
                 device='cuda'):

        super(GPT2, self).__init__()

        self.args = args 

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.device = device

        # Dropout 
        self.dropout = nn.Dropout(self.dropout)
        
        # Embeddings 
        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)     
        self.position_embeddings = nn.Embedding(max_seq_length, embedding_dim)
        
        # Transformer Blocks    
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(args, embedding_dim, num_attention_heads, embedding_dim * 4, max_seq_length, dropout)
            for _ in range(num_layers)
        ])

        # Final Layer Norm 
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Linear output layer 
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight
        
        # Initialize Weights 
        self.apply(self._init_weights)

        # Scaled Initialization for Residual Layers 
        for pn, p in self.named_parameters():
            if p.dim() > 1:
                if 'fc2' in pn or 'w_o' in pn:
                    p.data.normal_(mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

        self.name = f"GPT2_{num_layers}L_{num_attention_heads}H_{embedding_dim}D_{args.layer}"
        self.to(device)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, target=None):
        batch_size, seq_length = input_ids.size()

        # Create position ids 
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=self.device)

        # Embeddings 
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        x = token_embeds + position_embeds

        x = self.dropout(x)

        # Transformer Blocks 
        for layer in self.transformer_blocks:
            x = layer(x)

        # Final Layer Norm 
        x = self.layer_norm(x)

        if target is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None 

        return logits, loss
    
    def parameter_count(self, non_embeddings=True): 
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, trainable_params
    
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_embeddings, num_heads, max_seq_length, dropout):
        super(CausalMultiHeadAttention, self).__init__()

        assert d_embeddings % num_heads == 0, "Match Embeddings with Number of Heads"

        self.d_embedding = d_embeddings
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.d_heads = d_embeddings // num_heads


        self.w_k = nn.Linear(d_embeddings, d_embeddings)
        self.w_q = nn.Linear(d_embeddings, d_embeddings)
        self.w_v = nn.Linear(d_embeddings, d_embeddings)
        self.w_o = nn.Linear(d_embeddings, d_embeddings)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length)
        self.register_buffer('causal_mask', causal_mask)

    def split_heads(self, x):
        batch_size, seq_length, d_embeddings = x.shape 
        return x.view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_heads = x.shape 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * d_heads)        
        
    def forward(self, x):
        k = self.split_heads(self.w_k(x))
        q = self.split_heads(self.w_q(x))
        v = self.split_heads(self.w_v(x))        

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(v.size(-1)))

        seq_length = attn_scores.size(-2)
        mask_slice = self.causal_mask[:, :, :seq_length, :seq_length]
        attn_scores = attn_scores.masked_fill(mask_slice == 0, float('-inf'))

        # Softmax and weighting 
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = self.combine_heads(attn_output)
        attn_output = self.w_o(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output

class CausalMultiHeadConvNNAttention(nn.Module):
    def __init__(self, d_embeddings, num_heads, max_seq_length, dropout, K, convolution_type="depthwise"):
        super(CausalMultiHeadConvNNAttention, self).__init__()
        assert d_embeddings % num_heads == 0, "Match Embeddings with Number of Heads"
        assert K > 0 and K <= d_embeddings // num_heads, "K must be between 1 and max_seq_length"

        self.d_embeddings = d_embeddings
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.d_heads = d_embeddings // num_heads
        self.K = K
        self.convolution_type = convolution_type

        self.w_k = nn.Linear(d_embeddings, d_embeddings)
        self.w_q = nn.Linear(d_embeddings, d_embeddings)
        self.w_v = nn.Linear(d_embeddings, d_embeddings)
        self.w_o = nn.Linear(d_embeddings, d_embeddings)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.in_channels = d_embeddings // num_heads
        self.out_channels = d_embeddings // num_heads

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

        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length)
        self.register_buffer('causal_mask', causal_mask)

    def split_heads(self, x):
        batch_size, seq_length, d_embeddings = x.shape 
        return x.view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_heads = x.shape 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, num_heads * d_heads)
    
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
        q = self.split_heads(self.w_q(x)) # (B, NH, SL, DK)
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
       

        # Attention Matrix: (B, NH, SL, SL) - Q @ K^T
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_heads)
        seq_length = attn_matrix.size(-2)
        mask_slice = self.causal_mask[:, :, :seq_length, :seq_length]
        attn_matrix = attn_matrix.masked_fill(mask_slice == 0, float('-inf'))

        # Merge B and num_heads into dim for prime & conv 
        ## (B, NH, SL, DK) → (B*NH, DK, SL) for v and (B, NH, SL, SL) → (B*NH, SL, SL)
        v_merged = v.reshape(B * self.num_heads, seq_length, self.d_heads).permute(0, 2, 1)
        am_merged = attn_matrix.reshape(B * self.num_heads, seq_length, seq_length)
        
        # Prime and Convolution 
        prime = self._prime(v_merged, am_merged, self.K)
        out = self.conv(prime) # (B*num_heads, d_k, seq_length) 

        # Reshape back: (B*NH, DK, SL) → (B, NH, SL, DK)
        out = out.permute(0, 2, 1).contiguous().view(B, self.num_heads, seq_length, self.d_heads)
        out = self.attn_dropout(out) 

        # Combine Heads and Final Linear Projection
        attn_output = self.w_o(self.combine_heads(out)) # (B, SL, d_hidden)

        attn_output = self.resid_dropout(attn_output)
        return attn_output

# 1. TRITON KERNEL (FORWARD PASS)
@triton.jit
def fused_prime_conv_fwd_kernel(
    # Pointers to matrices
    v_ptr, indices_ptr, values_ptr, weight_ptr, out_ptr,
    # Strides to handle memory layout
    stride_vb, stride_vt, stride_vd,
    stride_ib, stride_it, stride_ik,
    stride_wb, stride_wk,
    stride_ob, stride_ot, stride_od,
    # Matrix dimensions
    B_NH, T, D: tl.constexpr, K: tl.constexpr,
    # Meta-parameters
    BLOCK_D: tl.constexpr
):
    """
    Fuses the gathering of V, multiplication by Top-K attention weights, 
    and the depthwise convolution step.
    """
    pid_b_t = tl.program_id(0) # 1D grid covering Batch*Heads and Seq_len
    pid_b = pid_b_t // T
    pid_t = pid_b_t % T

    # Set up channel offsets
    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # Initialize accumulator for the convolution sum
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over the Top-K elements
    for k in range(K):
        # 1. Load the index and attention value for the k-th top element
        idx_offset = pid_b * stride_ib + pid_t * stride_it + k * stride_ik
        v_idx = tl.load(indices_ptr + idx_offset)
        attn_val = tl.load(values_ptr + idx_offset)

        # 2. Load the V vector for all D channels at the gathered index
        v_offsets = pid_b * stride_vb + v_idx * stride_vt + d_offsets * stride_vd
        v_vec = tl.load(v_ptr + v_offsets, mask=mask_d, other=0.0)

        # 3. Load the Depthwise Convolution weight for this K step
        w_offsets = d_offsets * stride_wb + k * stride_wk
        w_vec = tl.load(weight_ptr + w_offsets, mask=mask_d, other=0.0)

        # 4. Multiply and accumulate (Gather * Attn_Value * Conv_Weight)
        acc += v_vec * attn_val * w_vec
        
    # Cast the fp32 accumulator back to the pointer's original dtype (fp16, bf16, or fp32)
    acc_casted = acc.to(v_ptr.dtype.element_ty) 
    
    # Store the final convolved output
    out_offsets = pid_b * stride_ob + pid_t * stride_ot + d_offsets * stride_od
    tl.store(out_ptr + out_offsets, acc_casted, mask=mask_d)


# 2. AUTOGRAD WRAPPER (FORWARD + BACKWARD)
class FusedPrimeConvFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda') # Tell AMP to let this pass through natively
    def forward(ctx, v, topk_indices, topk_values, conv_weight):
        # Save tensors needed for the backward pass
        ctx.save_for_backward(v, topk_indices, topk_values, conv_weight)
        
        B_NH, T, D = v.shape
        _, _, K = topk_indices.shape
        
        # Ensure contiguous memory for predictable strides
        v = v.contiguous()
        topk_indices = topk_indices.contiguous()
        topk_values = topk_values.contiguous()
        weight = conv_weight.squeeze(1).contiguous() # Squeeze from (D, 1, K) to (D, K)
        
        out = torch.empty_like(v)
        
        # Grid computes one block per query token per batch/head
        grid = lambda meta: (B_NH * T, )
        BLOCK_D = triton.next_power_of_2(D)
        
        fused_prime_conv_fwd_kernel[grid](
            v, topk_indices, topk_values, weight, out,
            v.stride(0), v.stride(1), v.stride(2),
            topk_indices.stride(0), topk_indices.stride(1), topk_indices.stride(2),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            B_NH, T, D, K,
            BLOCK_D=BLOCK_D
        )
        return out

    @staticmethod
    @custom_bwd(device_type='cuda') # Tells AMP how to handle the backward pass
    def backward(ctx, grad_out):
        v, topk_indices, topk_values, conv_weight = ctx.saved_tensors
        B_NH, T, D = v.shape
        _, _, K = topk_indices.shape
        weight = conv_weight.squeeze(1) # (D, K)
        
        grad_out = grad_out.contiguous()

        # Flatten indices to (B_NH, T*K, 1) and expand to D channels
        idx_flat = topk_indices.view(B_NH, T * K, 1).expand(-1, -1, D)
        
        # Gather V directly to shape (B_NH, T*K, D), then reshape
        v_gathered = torch.gather(v, 1, idx_flat).view(B_NH, T, K, D)

        # Pre-compute shapes for broadcasting and CAST to grad_out's dtype (fp16/bf16)
        target_dtype = grad_out.dtype
        grad_out_exp = grad_out.unsqueeze(2)                
        val_exp = topk_values.unsqueeze(-1).to(target_dtype)                 
        weight_t_exp = weight.t().unsqueeze(0).unsqueeze(0).to(target_dtype) 
        v_gathered = v_gathered.to(target_dtype)

        # Gradient w.r.t topk_values (dVal) - Cast back to original dtype (fp32)
        grad_val = (grad_out_exp * v_gathered * weight_t_exp).sum(dim=-1).to(topk_values.dtype) 

        # Gradient w.r.t conv_weight (dW) - Cast back to original weight dtype (fp32)
        grad_weight_raw = (grad_out_exp * v_gathered * val_exp).sum(dim=(0, 1)) 
        grad_weight = grad_weight_raw.t().unsqueeze(1).to(weight.dtype) 

        # Gradient w.r.t V (dV)
        dv_gathered = grad_out_exp * val_exp * weight_t_exp 
        grad_v = torch.zeros_like(v)
        
        # Now both are guaranteed to be target_dtype (e.g., float16)
        grad_v.scatter_add_(1, idx_flat, dv_gathered.view(B_NH, T * K, D))

        # topk_indices is discrete, so its gradient is None.
        return grad_v, None, grad_val, grad_weight


class FastCausalMultiHeadConvNNAttention(nn.Module):
    def __init__(self, d_embeddings, num_heads, max_seq_length, dropout, K, convolution_type="depthwise"):
        super(FastCausalMultiHeadConvNNAttention, self).__init__()
        assert d_embeddings % num_heads == 0, "Match Embeddings with Number of Heads"
        assert K > 0 and K <= d_embeddings // num_heads, "K must be between 1 and max_seq_length"

        self.d_embeddings = d_embeddings
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.d_heads = d_embeddings // num_heads
        self.K = K
        self.convolution_type = convolution_type

        self.w_k = nn.Linear(d_embeddings, d_embeddings)
        self.w_q = nn.Linear(d_embeddings, d_embeddings)
        self.w_v = nn.Linear(d_embeddings, d_embeddings)
        self.w_o = nn.Linear(d_embeddings, d_embeddings)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.conv_weight = nn.Parameter(torch.ones(self.d_heads, 1, self.K)) # (D, 1, K) for depthwise conv

        causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).view(1, 1, max_seq_length, max_seq_length)
        self.register_buffer('causal_mask', causal_mask)

    def split_head(self, x):
        batch_size, seq_length, d_hidden = x.size() 
        return x.view(batch_size, seq_length, self.num_heads, self.d_heads).transpose(1, 2)

    def forward(self, x):
        B = x.shape[0]

        # Linear Projection + Split Heads 
        q = self.split_head(self.w_q(x)) # (B, NH, SL, DK)
        k = self.split_head(self.w_k(x))
        v = self.split_head(self.w_v(x))

        # Attention Matrix: (B, NH, SL, SL)
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_heads)
        seq_length = attn_matrix.size(-2)
        mask_slice = self.causal_mask[:, :, :seq_length, :seq_length]
        attn_matrix = attn_matrix.masked_fill(mask_slice == 0, float('-inf'))   
        
        # Top-K Selection
        topk_values, topk_indices = torch.topk(attn_matrix, k=self.K, dim=-1, largest=True)
        topk_values = torch.softmax(topk_values, dim=-1)

        # Merge Batch and Heads for the Triton Kernel
        v_merged = v.reshape(B * self.num_heads, seq_length, self.d_heads)
        topk_indices_merged = topk_indices.reshape(B * self.num_heads, seq_length, self.K)
        topk_values_merged = topk_values.reshape(B * self.num_heads, seq_length, self.K)

        # Apply Fused Triton Operation (Forward + Backward handled automatically)
        out = FusedPrimeConvFunction.apply(
            v_merged, topk_indices_merged, topk_values_merged, self.conv_weight
        )

        # Reshape back: (B*NH, SL, DK) → (B, NH, SL, DK) → (B, SL, d_hidden)
        out = out.view(B, self.num_heads, seq_length, self.d_heads)
        out = out.transpose(1, 2).contiguous().view(B, seq_length, self.d_embeddings)
        out = self.attn_dropout(out) 

        # Final Linear Projection
        output = self.w_o(out)
        return output
    

class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(MLP, self).__init__() 
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()  # Default to GELU for the intermediate activation

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, args, d_model, num_heads, d_ff, max_seq_length, dropout):
        super(TransformerBlock, self).__init__()

        convnn_attn_params = {
            "K": args.K,
            "convolution_type": args.convolution_type,
        }

        convnn_attn_sampled_params = {
            "K": args.K, 
            "sampling_type": args.sampling_type,
            "num_samples": args.num_samples,
            "sample_padding": args.sample_padding,
            "convolution_type": args.convolution_type, 
        }

        
        if args.layer == "ConvNNAttention":
             self.attention = CausalMultiHeadConvNNAttention(d_model, num_heads, max_seq_length, dropout, **convnn_attn_params)
        elif args.layer == "FastConvNNAttention":
            self.attention = FastCausalMultiHeadConvNNAttention(d_model, num_heads, max_seq_length, dropout, **convnn_attn_params)
        elif args.layer == "Attention":           
            self.attention = CausalMultiHeadAttention(d_model, num_heads, max_seq_length, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-Norm Multi-Head Attention 
        norm_x = self.layer_norm1(x)
        attn_output = self.attention(norm_x)
        x = x + attn_output

        # Post-Norm MLP
        norm_x = self.layer_norm2(x)
        mlp_output = self.mlp(norm_x)
        x = x + mlp_output
        return x


if __name__ == "__main__":
    from types import SimpleNamespace
        
    args = SimpleNamespace(
        layer = "ConvNNAttention",
        K=10,
        convolution_type="depthwise",
        sampling_type="all",
        num_samples=-1,
        sample_padding=0
    )

    args2 = SimpleNamespace(
        layer = "Attention",
        K=10,
        convolution_type="depthwise",
        sampling_type="all",
        num_samples=-1,
        sample_padding=0
    )


    print("Testing GPT2 with ConvNN Attention Layer")
    convnn_gpt = GPT2(args, device='cpu')
    total_params = convnn_gpt.parameter_count()
    print(f"Total Parameters: {total_params}")

    ex = torch.randint(0, 50257, (2, 1024)).long()
    logits, loss = convnn_gpt(ex)
    print("Logits shape:", logits.shape)

    print() 
    print("Testing GPT2 with Standard Attention Layer")
    attn_gpt = GPT2(args2, device='cpu')
    total_params = attn_gpt.parameter_count()
    print(f"Total Parameters: {total_params}")

    ex = torch.randint(0, 50257, (2, 1024)).long()
    logits, loss = attn_gpt(ex)
    print("Logits shape:", logits.shape)

    # Sanity Check -> When K = seq_length (K=10 for 1024 seq_length and 12 heads)
    convnn_gpt.eval()
    attn_gpt.eval()
    convnn = convnn_gpt.transformer_blocks[0].attention
    attn = attn_gpt.transformer_blocks[0].attention

    # Copy all weights
    convnn.w_k.weight.data.copy_(attn.w_k.weight.data)
    convnn.w_k.bias.data.copy_(attn.w_k.bias.data)
    convnn.w_q.weight.data.copy_(attn.w_q.weight.data)
    convnn.w_q.bias.data.copy_(attn.w_q.bias.data)
    convnn.w_v.weight.data.copy_(attn.w_v.weight.data)
    convnn.w_v.bias.data.copy_(attn.w_v.bias.data)
    convnn.w_o.weight.data.copy_(attn.w_o.weight.data)
    convnn.w_o.bias.data.copy_(attn.w_o.bias.data)
    convnn.conv.weight.data.fill_(1.0)

    # Test the attention modules directly on the same input
    # Use a float tensor as if it were already embedded
    ex = torch.randn(2, 10, 768)  # (B, SL, d_model)
    with torch.no_grad():
        convnn_out = convnn(ex)
        attn_out = attn(ex)

    print("Difference:", torch.abs(convnn_out - attn_out).mean().item())
    print("allclose:", torch.allclose(convnn_out, attn_out, atol=1e-5))