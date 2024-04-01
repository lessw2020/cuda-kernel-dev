#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


/* 1. Load From SMEM to GMEM */ 

/*  2. Load From SMEM to Registers */


// 3. MMA Operation
__global__ void mma(
    uint32_t const *a_frag,
    uint32_t const *b_frag,
    uint32_t const *c_frag) {

        // Compute the coordinates of accesses to A and B Matrices
        int outer = threadIdx.x / 4;
        int inner = threadIdx.x % 4;

        /// Compute linear offsets for the accumulator matrices
        int c_row = threadIdx.x / 4;
        int c_col = 2 * (threadIdx.x % 4);

        // Compute linear offsets into each matrix
        int ab_idx = outer * 4 + inner;
        int cd_idx = c_row * 8 + c_col;

        // Issue Tensor Core Operation
        asm volatile(
            "mma.sync.aligned.m8n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7 }, {%8, %9}, {%10, %11, %12, %13}; \n"
            : "=f"(c[0]),"=f"(c[0]), "=f"(c[0]), "=f"(c[0]), "=f"(c[0])
            : "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
              "f" (c[0]), "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}