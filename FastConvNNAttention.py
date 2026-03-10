import torch
import torch.nn as nn
import numpy as np
import triton
import triton.language as tl
from torch.amp import custom_fwd, custom_bwd
# ==========================================
# 1. TRITON KERNEL (FORWARD PASS)
# ==========================================
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


# ==========================================
# 2. AUTOGRAD WRAPPER (FORWARD + BACKWARD)
# ==========================================
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


# ==========================================
# 3. PYTORCH MODULE
# ==========================================
class FastMultiHeadConvNNAttention(nn.Module):
    def __init__(self, 
                 d_hidden,
                 num_heads, 
                 attention_dropout, 
                 K,
                 convolution_type="depthwise",
                 seq_length=197):

        super(FastMultiHeadConvNNAttention, self).__init__()
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

        # Depthwise Convolution weights matching PyTorch's native Conv1d shape
        self.conv_weight = nn.Parameter(torch.ones(self.d_k, 1, self.K))

    def split_head(self, x):
        batch_size, seq_length, d_hidden = x.size() 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        B = x.shape[0]

        # Linear Projection + Split Heads 
        q = self.split_head(self.W_q(x)) # (B, NH, SL, DK)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))

        # Attention Matrix: (B, NH, SL, SL)
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Top-K Selection
        topk_values, topk_indices = torch.topk(attn_matrix, k=self.K, dim=-1, largest=True)
        topk_values = torch.softmax(topk_values, dim=-1)

        # Merge Batch and Heads for the Triton Kernel
        v_merged = v.reshape(B * self.num_heads, self.seq_length, self.d_k)
        topk_indices_merged = topk_indices.reshape(B * self.num_heads, self.seq_length, self.K)
        topk_values_merged = topk_values.reshape(B * self.num_heads, self.seq_length, self.K)

        # Apply Fused Triton Operation (Forward + Backward handled automatically)
        out = FusedPrimeConvFunction.apply(
            v_merged, topk_indices_merged, topk_values_merged, self.conv_weight
        )

        # Reshape back: (B*NH, SL, DK) → (B, NH, SL, DK) → (B, SL, d_hidden)
        out = out.view(B, self.num_heads, self.seq_length, self.d_k)
        out = out.transpose(1, 2).contiguous().view(B, self.seq_length, self.d_hidden)
        out = self.dropout(out) 

        # Final Linear Projection
        output = self.W_o(out)
        return output


# ==========================================
# 4. QUICK VERIFICATION TEST
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type != "cuda":
        print("Warning: Triton requires a CUDA-enabled GPU. This test will fail on CPU.")
    else:
        # Hyperparameters matching standard ViT-Base
        BATCH_SIZE = 2
        SEQ_LENGTH = 197
        D_HIDDEN = 768
        NUM_HEADS = 12
        K = 8
        
        print("Initializing FastMultiHeadConvNNAttention...")
        model = FastMultiHeadConvNNAttention(
            d_hidden=D_HIDDEN, 
            num_heads=NUM_HEADS, 
            attention_dropout=0.1, 
            K=K, 
            seq_length=SEQ_LENGTH
        ).to(device)
        
        # Dummy input
        x = torch.randn(BATCH_SIZE, SEQ_LENGTH, D_HIDDEN, device=device, requires_grad=True)
        
        print("Running Forward Pass...")
        out = model(x)
        print(f"Output shape: {out.shape} (Expected: [{BATCH_SIZE}, {SEQ_LENGTH}, {D_HIDDEN}])")
        
        print("Running Backward Pass...")
        loss = out.sum()
        loss.backward()
        print(f"Input gradient shape: {x.grad.shape}")
        print("Success! Forward and backward passes completed.")