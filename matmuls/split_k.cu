#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Square Matrices
// void split_k_kernel_launcher(float* A, float* B, float* C, int N) {

//     // Kernel launch parameters
//     dim3 blocks(16, 16, 1);
//     dim3 threads(16, 16, 1);

//     split_k_kernel<<<blocks, threads>>>(A, B, C, N);
// }



// GEMM (M, N and K)

// __global__ void split_k_kernel(float *A, float *B, float *C, int M, int N, int K) {
    
//     // Global Row and Column Indices
//     int col = blockIdx.y * blockDim.y + threadIdx.y;
//     int row = blockIdx.x * blockDim.x + threadIdx.x;

//     // Boundary Check
//     if (row < M && col < N) {  
//         float acc = 0.0;
//         for (int k = 0; k < N; ++k) {
//             acc += A[row * M + k] * B[k * N + col];
//         }
//         C[row * N + col] = acc;
//     }
// }


// void split_k_kernel_launcher(float* A, float* B, float* C, int M, int N, int K) {

//     // Kernel launch parameters
//     dim3 threads(16, 16, 1);

//     int thread_blocks_x = (N + threads.x - 1) / threads.x; // makes sure there are enough threads to cover if problem size
//     int thread_blocks_y = (M + threads.y - 1) / threads.y; // is not divisible by 16 equivalent to ceil()

//     dim3 blocks(thread_blocks_x, thread_blocks_y, 1);
//     split_k_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
// }

// Tiled Matrix Multiplication
// #define block_m 16
// #define block_n 32
// #define block_k 128

// __global__ void split_k_kernel(float *A, float *B, float *C, int m, int n, int k) {

//     __shared__ int As[block_m][block_k];
//     __shared__ int Bs[block_k][block_n];

//     int tx = threadIdx.x; 
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;  
//     int by = blockIdx.y;

//     // Global Row and Column Indices
//     int row = by * blockDim.y + ty;
//     int col = bx * blockDim.x + tx;
    
//     float acc = 0;

//     for (int t=0; t < n/block_n; ++t){
//         /* 
//         Every Thread in a ThreadBlock loads one element into shared memory
//         The element location in shared memory corresponds to the thread's position
//         in the threadblock (e.g. thread [0,0] loads for A[0 * TILE_SIZE_* + 0])
//         Indexing parameters:
//         A: 
//                 1. row * m: Indexes the global row for this thread (loop invariant)
//                 2. i * TILE_SIZE_K: Indexes the new set of columns each iteration
//                 3. tx: Indexes the column within the set
//         B:
//                 k * TILE_SIZE_K * N: Indexes the next set of rows each iteration
//                 ty*n: Indexes the row within the set 
//                 col: Indexes the global column (loop-invariant)
//         */
//         // Collaborative loading of A and B tiles into shared memory
//         As[ty][tx] = A[row*n + t*block_k];
//         Bs[ty][tx] = B[(t*block_k+ty)*k + col];

//         // Ensure all threads have loaded their data before proceeding
//         __syncthreads();
//         for (int i = 0; i < block_k; i++){
//             /*
//             1. The row in A is given by ty 
//             2. The column in B is given by tx
//             3. Iteration Dimension is inner dimension K
//             */
//             acc += As[ty][i] * Bs[i][tx];
//         }
//         // Ensure some threads don't progress and overwrite current 
//         // shared memory values
//         __syncthreads();
//         // Write out to DRAM
//         C[(row * k) + col] = acc;
//     }
// }


// void split_k_kernel_launcher(float* A, float* B, float* C, int m, int n, int k) {

//     // Kernel launch parameters
//     dim3 threads(16, 16, 1);

//     int thread_blocks_x = (n + threads.x - 1) / threads.x; // makes sure there are enough threads to cover if problem size
//     int thread_blocks_y = (m + threads.y - 1) / threads.y; // is not divisible by 16 equivalent to ceil()

//     dim3 blocks(thread_blocks_x, thread_blocks_y, 1);
//     split_k_kernel<<<blocks, threads>>>(A, B, C, m, n, k);
// }




// Advanced Boundary Condition Handling

// #define cdiv(M, N) (((M) + (N)-1) / (N))

// #define block_m 16
// #define block_n 32
// #define block_k 128

// __global__ void split_k_kernel(float *A, float *B, float *C, int m, int n, int k) {
//     __shared__ float As[block_m][block_k];
//     __shared__ float Bs[block_k][block_n];

//     int tx = threadIdx.x; 
//     int ty = threadIdx.y;
//     int bx = blockIdx.x;  
//     int by = blockIdx.y;

//     int row = by * blockDim.y + ty;
//     int col = bx * blockDim.x + tx;
    
//     float acc = 0;
//     for (int t = 0; t < (k + block_k - 1) / block_k; t++) {
//         if (row < m && t*block_k + tx < k) // Make sure we don't read out of bounds
//             As[ty][tx] = A[row*k + t*block_k + tx];
//         else
//             As[ty][tx] = 0.0;

//         if (col < n && t*block_k + ty < k) // Make sure we don't read out of bounds
//             Bs[ty][tx] = B[(t*block_k + ty)*n + col];
//         else
//             Bs[ty][tx] = 0.0;
//         __syncthreads();

//         for (int i = 0; i < block_k; i++) {
//             acc += As[ty][i] * Bs[i][tx];
//         }
//         __syncthreads();
//     }
//     if (row < m && col < n) // Make sure we don't write out of bounds
//         C[(row * n) + col] = acc;
// }

// Tiled GEMM
#define tile_size 16
#define cdiv(M, N) (((M) + (N)-1) / (N))

__global__ void split_k_kernel_v1(float *A, float *B, float *C, int *perm, int m, int n, int k) {

    __shared__ float As[tile_size][tile_size];
    __shared__ float Bs[tile_size][tile_size];

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    
    int bx = blockIdx.x;  
    int by = blockIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float acc = 0;
    for (int t=0; t < n/tile_size; ++t){

        int global_index = row*n + t*tile_size + tx;
        int global_col = global_index % k;
        int global_row = global_index / k;
        
        int perm_col = perm[global_col];
        int perm_idx = global_row*k + perm_col;

        As[ty][tx] = A[perm_idx];
        Bs[ty][tx] = B[(t*tile_size+ty)*k + col];

        __syncthreads();

        for (int i = 0; i < tile_size; i++){
            acc += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    C[row*k + col] = acc;
}

void split_k_kernel_launcher(float* A, float* B, float* C, int* perm, int m, int n, int k) {

    dim3 threads(tile_size, tile_size, 1);

    int thread_blocks_m = cdiv(m, tile_size);
    int thread_blocks_n = cdiv(n, tile_size);

    printf("thread_blocks_m: %i \n", thread_blocks_m);
    printf("thread_blocks_n: %i \n", thread_blocks_n);

    dim3 blocks(thread_blocks_m, thread_blocks_n, 1);
    split_k_kernel_v1<<<blocks, threads>>>(A, B, C, perm, m, n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel launch: " << cudaGetErrorString(err) << std::endl;
    }
}
