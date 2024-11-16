from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl


# Previous configurations remain the same...
@dataclass
class H100VectorConfig:
    """H100-optimized configuration with enhanced vectorization"""

    block_size_m: int = 256
    block_size_k: int = 256
    min_scale: float = 1e-4
    dtype: torch.dtype = torch.float8_e4m3fn
    vec_size: int = 32
    num_warps: int = 8
    num_stages: int = 8


# Constants
FP8_E4M3_MAX: tl.constexpr = 448.0


# Kernel remains the same...
@triton.jit
def kernel_block_fp8_quantize_h100_vectorized(
    A,
    A_scale,
    A_fp8,
    M,
    K,
    stride_am,
    stride_ak,
    stride_ascale_m,
    stride_ascale_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
) -> None:
    """
    vectorized traversal of block
    """
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k

    # Base offsets for block
    offs_m_base = block_m * BLOCK_M
    offs_k_base = block_k * BLOCK_K

    # Initialize single maximum value
    block_max = -float("inf")

    # Create vector range once
    vec_range = tl.arange(0, VEC_SIZE)

    # Process block with vectorized operations
    for m_idx in range(0, BLOCK_M, VEC_SIZE):
        m_offs = offs_m_base + m_idx + vec_range
        m_mask = m_offs < M

        for k_idx in range(0, BLOCK_K, VEC_SIZE):
            k_offs = offs_k_base + k_idx + vec_range
            k_mask = k_offs < K

            offs_a = m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask = m_mask[:, None] & k_mask[None, :]

            a_block = tl.load(A + offs_a, mask=mask, other=0.0)
            block_max = tl.maximum(block_max, tl.max(tl.abs(a_block)))

    # Compute scale with numerical stability
    scale = FP8_E4M3_MAX / tl.maximum(block_max, MIN_SCALE)

    # Store scale
    scale_offset = block_m * stride_ascale_m + block_k * stride_ascale_k
    tl.store(A_scale + scale_offset, scale)

    # Quantize with vectorized operations
    for m_idx in range(0, BLOCK_M, VEC_SIZE):
        m_offs = offs_m_base + m_idx + vec_range
        m_mask = m_offs < M

        for k_idx in range(0, BLOCK_K, VEC_SIZE):
            k_offs = offs_k_base + k_idx + vec_range
            k_mask = k_offs < K

            offs_a = m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
            mask = m_mask[:, None] & k_mask[None, :]

            a_block = tl.load(A + offs_a, mask=mask, other=0.0)
            a_fp8 = (a_block * scale).to(tl.float8e4nv)
            tl.store(A_fp8 + offs_a, a_fp8, mask=mask)


def get_optimal_h100_vector_config(M: int, K: int) -> Dict[str, int]:
    """Configuration function remains the same"""
    if M * K < 1024 * 1024:
        return {"BLOCK_M": 128, "BLOCK_K": 128, "VEC_SIZE": 32}
    elif M * K < 4 * 1024 * 1024:
        return {"BLOCK_M": 256, "BLOCK_K": 256, "VEC_SIZE": 32}
    else:
        return {"BLOCK_M": 256, "BLOCK_K": 256, "VEC_SIZE": 32}


def tensor_quantize_to_block_fp8_h100_vectorized(
    x: torch.Tensor, config: Optional[H100VectorConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    H100-optimized quantization with enhanced vectorization
    Returns:
        Tuple[Tensor, Tensor]: (quantized_tensor, scale_tensor)
        scale_tensor has shape (M//block_size_m, K//block_size_k)
    """
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")

    M, K = x.shape
    kernel_config = get_optimal_h100_vector_config(M, K)

    if config is None:
        config = H100VectorConfig()

    # x = x.contiguous()

    # Calculate grid dimensions
    gridm = triton.cdiv(M, kernel_config["BLOCK_M"])
    gridk = triton.cdiv(K, kernel_config["BLOCK_K"])

    # Allocate output tensors
    x_scale = torch.empty(
        (gridm, gridk),
        device=x.device,
        dtype=torch.float32,
        memory_format=torch.contiguous_format,
    )
    x_fp8_out = torch.empty(
        (M, K),
        device=x.device,
        dtype=config.dtype,
        memory_format=torch.contiguous_format,
    )

    kernel_block_fp8_quantize_h100_vectorized[(gridm * gridk,)](
        x,
        x_scale,
        x_fp8_out,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        BLOCK_M=kernel_config["BLOCK_M"],
        BLOCK_K=kernel_config["BLOCK_K"],
        MIN_SCALE=config.min_scale,
        VEC_SIZE=kernel_config["VEC_SIZE"],
    )

    return x_fp8_out, x_scale


def dequantize_fp8(
    quantized: torch.Tensor, scale: torch.Tensor, block_size_m: int, block_size_k: int
) -> torch.Tensor:
    """
    Dequantize FP8 tensor with proper scale handling
    """
    M, K = quantized.shape
    gridm = triton.cdiv(M, block_size_m)
    gridk = triton.cdiv(K, block_size_k)

    # Expand scale tensor to match input shape
    scale_expanded = torch.zeros((M, K), device=quantized.device, dtype=torch.float32)

    for i in range(gridm):
        for j in range(gridk):
            m_start = i * block_size_m
            m_end = min((i + 1) * block_size_m, M)
            k_start = j * block_size_k
            k_end = min((j + 1) * block_size_k, K)

            scale_expanded[m_start:m_end, k_start:k_end] = scale[i, j]

    # Dequantize
    return quantized.float() / scale_expanded


def verify_accuracy(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    kernel_config: Dict[str, int],
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Dict[str, float]:
    """
    Verify quantization accuracy with fixed dequantization
    """
    # Dequantize with proper scale handling
    dequantized = dequantize_fp8(
        quantized, scale, kernel_config["BLOCK_M"], kernel_config["BLOCK_K"]
    )

    # Compute metrics
    abs_error = torch.abs(original - dequantized)
    rel_error = abs_error / (torch.abs(original) + 1e-7)

    return {
        "max_abs_error": abs_error.max().item(),
        "mean_abs_error": abs_error.mean().item(),
        "max_rel_error": rel_error.max().item(),
        "mean_rel_error": rel_error.mean().item(),
    }


def benchmark_vectorized(
    M: int = 4096, K: int = 4096, num_warmup: int = 10, num_runs: int = 100
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Benchmark with fixed accuracy verification
    """
    device = torch.device("cuda")
    x = torch.randn(M, K, device=device)
    kernel_config = get_optimal_h100_vector_config(M, K)

    # Warmup
    for _ in range(num_warmup):
        tensor_quantize_to_block_fp8_h100_vectorized(x)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    x_fp8, x_scale = None, None
    for _ in range(num_runs):
        x_fp8, x_scale = tensor_quantize_to_block_fp8_h100_vectorized(x)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)

    avg_time = elapsed_time / num_runs
    elements_per_second = (M * K) / (avg_time * 1e-3)
    throughput_gb = elements_per_second * 4 / 1e9
    tflops = (M * K * 2) / (avg_time * 1e-3) / 1e12

    # Verify accuracy with fixed dequantization
    accuracy_metrics = verify_accuracy(x, x_fp8, x_scale, kernel_config)

    return throughput_gb, avg_time, tflops, accuracy_metrics


def main():
    """
    Run benchmarks with accuracy verification
    """
    sizes = [
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
        # (65536, 65536),
    ]

    print("\nH100 FP8 Quantization Benchmark with Enhanced Vectorization")
    print("-" * 80)
    print(
        f"{'Size':>12} {'Throughput':>12} {'Latency':>10} {'TFLOPS':>10} {'Max Rel Error':>15}"
    )
    print("-" * 80)

    for M, K in sizes:
        throughput, latency, tflops, accuracy = benchmark_vectorized(M, K)
        print(
            f"{M}x{K:>8} {throughput:>10.1f} GB/s {latency:>8.2f}ms "
            f"{tflops:>8.1f} {accuracy['max_rel_error']:>13.2e}"
        )


if __name__ == "__main__":
    main()
