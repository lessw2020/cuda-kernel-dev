from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import triton
import triton.language as tl


@dataclass
class H100OptimizedConfig:
    """H100-specific configuration"""

    block_size_m: int = 256
    block_size_k: int = 256
    min_scale: float = 1e-4
    dtype: torch.dtype = torch.float8_e4m3fn


# Constants
FP8_E4M3_MAX: tl.constexpr = 448.0


@triton.jit
def kernel_block_fp8_quantize_h100_optimized(
    A,
    A_scale,
    A_fp8,  # Pointers
    M,
    K,  # Dimensions
    stride_am,
    stride_ak,  # Input strides
    stride_ascale_m,  # Scale strides
    stride_ascale_k,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MIN_SCALE: tl.constexpr,
) -> None:
    """
    H100-optimized kernel
    """
    # Program ID and block computation
    pid = tl.program_id(0)
    grid_k = tl.cdiv(K, BLOCK_K)
    block_m = pid // grid_k
    block_k = pid % grid_k

    # Create block pointers with proper striding
    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = block_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Create mask
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)

    # Compute memory offset for loading
    offs = offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # Load data
    a = tl.load(A + offs, mask=mask, other=0.0)

    # Compute maximum absolute value for scaling
    max_val = tl.max(tl.abs(a))

    # Calculate and store scale
    scale = FP8_E4M3_MAX / tl.maximum(max_val, MIN_SCALE)
    scale_offset = block_m * stride_ascale_m + block_k * stride_ascale_k
    tl.store(A_scale + scale_offset, scale)

    # Quantize and store
    a_scaled = a * scale
    a_fp8 = a_scaled.to(tl.float8e4nv)
    tl.store(A_fp8 + offs, a_fp8, mask=mask)


def get_optimal_h100_config(M: int, K: int) -> Dict[str, int]:
    """
    Get optimal H100 configuration based on input size
    """
    if M * K < 1024 * 1024:
        return {
            "BLOCK_M": 128,
            "BLOCK_K": 128,
        }
    elif M * K < 4 * 1024 * 1024:
        return {
            "BLOCK_M": 256,
            "BLOCK_K": 256,
        }
    else:
        return {
            "BLOCK_M": 256,
            "BLOCK_K": 256,
        }


def tensor_quantize_to_block_fp8_h100_optimized(
    x: torch.Tensor, config: Optional[H100OptimizedConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    H100-optimized quantization function
    """
    if x.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got {x.dim()}D")

    # Get optimal configuration
    M, K = x.shape
    kernel_config = get_optimal_h100_config(M, K)

    if config is None:
        config = H100OptimizedConfig()

    # Ensure optimal memory layout
    x = x.contiguous()

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

    # Launch kernel
    kernel_block_fp8_quantize_h100_optimized[(gridm * gridk,)](
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


def benchmark_throughput(
    M: int = 4096, K: int = 4096, num_warmup: int = 10, num_runs: int = 100
) -> Tuple[float, float]:
    """
    Benchmark throughput and latency
    Returns (throughput GB/s, latency ms)
    """
    device = torch.device("cuda")
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(num_warmup):
        tensor_quantize_to_block_fp8_h100_optimized(x)

    # Timing
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_runs):
        tensor_quantize_to_block_fp8_h100_optimized(x)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)

    avg_time = elapsed_time / num_runs
    elements_per_second = (M * K) / (avg_time * 1e-3)
    throughput_gb = elements_per_second * 2 / 1e9  # 4 bytes per element

    return throughput_gb, avg_time


# Example usage with performance metrics
def main():
    # Test different sizes
    sizes = [(2048, 2048), (4096, 4096), (8192, 8192)]

    print("\nH100 FP8 Quantization Benchmark")
    print("-" * 40)

    for M, K in sizes:
        print(f"\nMatrix size: {M}x{K}")
        throughput, latency = benchmark_throughput(M, K)
        ops = M * K * 2  # FP8 conversion operations
        tflops = (ops / (latency * 1e-3)) / 1e12

        print(f"Throughput: {throughput:.2f} GB/s")
        print(f"Latency: {latency:.3f} ms")
        print(f"Performance: {tflops:.2f} TFLOPS")


if __name__ == "__main__":
    throughput, latency = benchmark_throughput(4096, 4096)
    print(f"Throughput: {throughput:.2f} GB/s")
    print(f"Latency: {latency:.3f} ms")
    main()
