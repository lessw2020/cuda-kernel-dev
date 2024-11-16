from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

# Constants
FP8_E4M3_MAX: tl.constexpr = 448.0  # Maximum value for e4m3 format


@dataclass
class QuantizationConfig:
    """Configuration for FP8 quantization"""

    block_size_m: int = 256
    block_size_k: int = 256
    min_scale: float = 1e-4  # Prevent division by zero
    dtype: torch.dtype = torch.float8_e4m3fn


@triton.jit
def kernel_block_fp8_quantize(
    A,  # Input tensor pointer
    A_scale,  # Scale tensor pointer
    A_fp8,  # Output FP8 tensor pointer
    M,
    K,  # Matrix dimensions
    stride_am,  # Stride for matrix A in M dimension
    stride_ak,  # Stride for matrix A in K dimension
    stride_ascale_m,  # Stride for scale tensor in M dimension
    stride_ascale_k,  # Stride for scale tensor in K dimension
    BLOCK_M: tl.constexpr,  
    BLOCK_K: tl.constexpr,  
    MIN_SCALE: tl.constexpr,  
) -> None:
    """
    Kernel for block-wise FP8 quantization with dynamic scaling
    """
    # Get program ID and compute block indices
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k

    # Compute row and column indices
    rm = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = block_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Compute memory offsets and masks
    a_offset = rm[:, None] * stride_am + rk[None, :] * stride_ak
    a_mask = (rm < M)[:, None] & (rk < K)[None, :]

    # Load input block
    a_block = tl.load(A + a_offset, mask=a_mask, other=0.0)

    # Compute scale factor with numerical stability
    max_abs = tl.maximum(tl.max(tl.abs(a_block)), MIN_SCALE)
    scale = FP8_E4M3_MAX / max_abs

    # Store scale factor
    scale_offset = block_m * stride_ascale_m + block_k * stride_ascale_k
    tl.store(A_scale + scale_offset, scale)

    # Quantize and store result
    a_fp8 = (a_block * scale).to(tl.float8e4nv)
    tl.store(A_fp8 + a_offset, a_fp8, mask=a_mask)


def tensor_quantize_to_block_fp8(
    x: torch.Tensor, config: Optional[QuantizationConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 with block-wise scaling.

    Args:
        x: Input tensor to quantize
        config: Quantization configuration (optional)

    Returns:
        Tuple of (quantized tensor, scale tensor)
    """
    if config is None:
        config = QuantizationConfig()

    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")

    # Get dimensions and compute grid size
    M, K = x.shape
    gridm = triton.cdiv(M, config.block_size_m)
    gridk = triton.cdiv(K, config.block_size_k)

    # Allocate output tensors
    x_scale = torch.empty((gridm, gridk), device=x.device, dtype=torch.float32)
    x_fp8_out = torch.empty((M, K), device=x.device, dtype=config.dtype)

    # Launch kernel
    kernel_block_fp8_quantize[(gridm * gridk,)](
        x,
        x_scale,
        x_fp8_out,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        BLOCK_M=config.block_size_m,
        BLOCK_K=config.block_size_k,
        MIN_SCALE=config.min_scale,
    )

    return x_fp8_out, x_scale


# Example usage
def quantize_example():
    # Create sample tensor
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)

    # Configure quantization
    config = QuantizationConfig(
        block_size_m=128, block_size_k=128, min_scale=1e-4, dtype=torch.float8_e4m3fn
    )

    # Quantize tensor
    x_fp8, x_scale = tensor_quantize_to_block_fp8(x, config)
    return x_fp8, x_scale


if __name__ == "__main__":
    quant_tensor, scale_tensor = quantize_example()
    print(f"Quantized tensor: {quant_tensor}")
    print(f"Scale tensor: {scale_tensor}")
