import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import numpy as np
import torch

@triton.jit
def gemm_kernel_tma(a_desc_ptr, b_desc_ptr, c_desc_ptr,
                    a_scale_desc_ptr, b_scale_desc_ptr,
                    prob_m, prob_n, prob_k, 
                    BLOCK_M: tl.constexpr, 
                    BLOCK_N: tl.constexpr, 
                    BLOCK_K: tl.constexpr,
                    BLOCK_SCALE: tl.constexpr,):
    
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(prob_m, BLOCK_M)
    num_pid_k = tl.cdiv(prob_k, BLOCK_K)
    
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N

    offs_scale = 0
    offs_k = 0

    scale_a = tl._experimental_descriptor_load(a_scale_desc_ptr, [offs_scale, offs_scale], [BLOCK_SCALE, BLOCK_SCALE], tl.float32)
    scale_b = tl._experimental_descriptor_load(b_scale_desc_ptr, [offs_scale, offs_scale], [BLOCK_SCALE, BLOCK_SCALE], tl.float32)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, num_pid_k):

        a = tl._experimental_descriptor_load(a_desc_ptr, [offs_am, offs_k], [BLOCK_M, BLOCK_K], tl.float8e4nv)
        b = tl._experimental_descriptor_load(b_desc_ptr, [offs_k, offs_bn], [BLOCK_K, BLOCK_N], tl.float8e4nv)

        accumulator = tl.dot(a, b.T, acc=accumulator, out_dtype=tl.float32)
        offs_k += BLOCK_K

    accumulator = scale_a * scale_b * accumulator
    accumulator = accumulator.to(tl.float16)
    tl._experimental_descriptor_store(c_desc_ptr, accumulator, [offs_am, offs_bn])


def matmul(a, b, scale_a, scale_b, config=None):

    M, _ = a.shape
    K, N = b.shape

    if config:
        BLOCK_M = config["BLOCK_M"]
        BLOCK_N = config["BLOCK_N"]
        BLOCK_K = config["BLOCK_K"]
        BLOCK_SCALE = config["BLOCK_SCALE"]
        num_warps = config["num_warps"]
        num_stages = config["num_stages"]
    
    else:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        BLOCK_SCALE = 64
        num_warps = 4
        num_stages = 3

    c = torch.empty((m, n), dtype=torch.float16, device='cuda')

    desc_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(a.data_ptr(), M, K, BLOCK_M, BLOCK_K, a.element_size(),
                                                            )
    desc_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(b.data_ptr(), K, N, BLOCK_K, BLOCK_N, b.element_size(),
                                                            )
    desc_c = triton.tools.experimental_descriptor.create_2d_tma_descriptor(c.data_ptr(), M, N, BLOCK_M, BLOCK_N, c.element_size(),
                                                            )
    
    desc_scale_a = triton.tools.experimental_descriptor.create_2d_tma_descriptor(scale_a.data_ptr(), BLOCK_SCALE, BLOCK_SCALE, BLOCK_SCALE, BLOCK_SCALE, scale_a.element_size())
    
    desc_scale_b = triton.tools.experimental_descriptor.create_2d_tma_descriptor(scale_b.data_ptr(), BLOCK_SCALE, BLOCK_SCALE, BLOCK_SCALE, BLOCK_SCALE,  scale_b.element_size(),
                                                        )
    
    total_blocks_m = triton.cdiv(M, BLOCK_M)
    total_blocks_n = triton.cdiv(N, BLOCK_N)
    
    grid = (total_blocks_m * total_blocks_n, 1, 1)
    k = gemm_kernel_tma[grid](
        desc_a, desc_b, desc_c,
        desc_scale_a, desc_scale_b,
        M, N, K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        BLOCK_SCALE,
        #num_warps=num_warps,
        #num_stages=num_stages,
    )
    return c

if __name__ == '__main__':

    m, k, n = 4096, 4096, 4096

    a = torch.randn((m, k), dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn)
    b = torch.randn((n, k), dtype=torch.float16, device='cuda').to(torch.float8_e4m3fn).t()

    scale_a = torch.ones((64, 64)).to(dtype=torch.float32, device='cuda') 
    scale_b = torch.ones((64, 64)).to(dtype=torch.float32, device='cuda')

    scale_a_ref = torch.ones((1,)).to(dtype=torch.float32, device='cuda') 
    scale_b_ref = torch.ones((1, )).to(dtype=torch.float32, device='cuda')

    ref = torch._scaled_mm(a, b, scale_a=scale_a_ref, scale_b=scale_b_ref, out_dtype=torch.float16)
    c = matmul(a, b, scale_a=scale_a, scale_b=scale_b)
    
    print(f"{ref=}")
    print(f"{c=}")

    torch.testing.assert_close(ref, c, atol=1e-2, rtol=0)