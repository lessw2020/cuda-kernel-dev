import torch
from torch.testing import assert_close
import os
import triton
import triton.language as tl
import time
from torch.profiler import profile, record_function, ProfilerActivity

os.environ['ENABLE_TMA'] = '1'

@triton.jit()
def grouped_launch(pid,
                   m, n,
                   block_m: tl.constexpr, block_n: tl.constexpr, group_m: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)

    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size

    return pid_m, pid_n


@triton.jit()
def col_major(pid,
              m, n,
              block_m: tl.constexpr, block_n: tl.constexpr):
    
    grid_m = tl.cdiv(m, block_m)    
    grid_n = tl.cdiv(n, block_n)
    
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    return pid_m, pid_n


@triton.jit
def gemm_block_ptr_splitk(a_ptr, b_ptr, c_ptr,  
                m, n, k,  
                stride_am, stride_ak,  
                stride_bk, stride_bn,  
                stride_cm, stride_cn,
                fp8_fast_accum, splitk, 
                block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,
                group_m: tl.constexpr  
                ):
    
    pid = tl.program_id(axis=0)
    pid_m, pid_n = grouped_launch(pid,
                                  m, n,
                                  block_m, block_n, group_m)
    pid_k = tl.program_id(1)
    grid_k = tl.cdiv(k, block_k*splitk)
    
    block_offset_m = pid_m * block_m
    block_offset_n = pid_n * block_n

    a_tile_ptr = tl.make_block_ptr(base=a_ptr, shape=(m, k), strides=(stride_am, stride_ak),
                                offsets=(block_offset_m, 0), block_shape=(block_m, block_k), order=(1, 0))
    
    b_tile_ptr = tl.make_block_ptr(base=b_ptr, shape=(k, n), strides=(stride_bk, stride_bn),
                                offsets=(0, block_offset_n), block_shape=(block_k, block_n), order=(0, 1))
    
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k in range(0, grid_k):

        a = tl.load(a_tile_ptr)
        b = tl.load(b_tile_ptr)

        if fp8_fast_accum:
            accumulator = tl.dot(a, b, accumulator, out_dtype=tl.float32)
        else: 
            accumulator += tl.dot(a, b, accumulator)

        a_tile_ptr = tl.advance(a_tile_ptr, [0, block_k*splitk])
        b_tile_ptr = tl.advance(b_tile_ptr, [block_k*splitk, 0])

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(m, n), strides=(stride_cm, stride_cn),
                                    offsets=(block_offset_m, block_offset_n), block_shape=(block_m, block_n),
                                    order=(1, 0))
    
    tl.store(c_block_ptr, accumulator.to(tl.float16))

def matmul(a, b):

    m, _ = a.shape
    k, n = b.shape

    block_m = 64
    block_n = 64
    block_k = 512
    group_m = 8
    num_warps = 8
    num_stages = 4
    splitk = 4

    fp8_fast_accum = True
    c = torch.empty((m, n), device=a.device, dtype=torch.float16)

    total_blocks_m = triton.cdiv(m, block_m)
    total_blocks_n = triton.cdiv(n, block_n)

    grid = (total_blocks_m*total_blocks_n, splitk,)
    gemm_block_ptr_splitk[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,  
        m=m, n=n, k=k,  
        stride_am=a.stride(0), stride_ak=a.stride(1),  
        stride_bk=b.stride(0), stride_bn=b.stride(1),  
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        fp8_fast_accum=fp8_fast_accum, 
        block_m=block_m, block_n=block_n, block_k=block_k, group_m=group_m,
        splitk=splitk,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


if __name__ == "__main__":

    torch.cuda.manual_seed(0)

    a_ = torch.zeros((16, 4096), device='cuda', dtype=torch.float8_e4m3fn)
    b_ = torch.zeros((4096, 4096), device='cuda', dtype=torch.float8_e4m3fn).T
    
    start = time.time()
    c = torch._scaled_mm(a_, b_, out_dtype=torch.float16, use_fast_accum=True)
    stop = time.time()
    print(f"cuBLAS FP8 {stop-start}\n")

    start = time.time()
    c = matmul(a_, b_)
    stop = time.time()
    print(f"Triton FP8 {stop-start}\n")

    a = torch.zeros((16, 4096), device='cuda', dtype=torch.float16)
    b = torch.zeros((4096, 4096), device='cuda', dtype=torch.float16).T

    # CAST to FP8 after? How to Verify Answer? 
    # Ask Naigang
    
    start = time.time()
    c = torch.matmul(a, b)
    stop = time.time()

    print(f"Triton FP16 {stop-start}\n")
    # print(c)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #           record_shapes=True,
    #           with_stack=True,
    #           profile_memory=True) as prof:

    #           split_k_output = matmul(a_, b_)
    #           torch_output = torch.matmul(a, b)
    #           cu_fp8 = torch._scaled_mm(a_, b_, out_dtype=torch.float16, use_fast_accum=True)

    # prof.export_chrome_trace("fp8_kernels.json")