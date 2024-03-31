//practice
__global__ void MatrixMatMul (float* M, float* N, float* P, int Width) {
    __shared__ float subTileM[Tile_Width][Tile_Width]
    __shared__ float subTileN[Tile_Width][Tile_Width]

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // identify P value to work on
    int Row = by * Tile_Width + ty;
    int Col = bx * Tile_Width + tx;

    float Pvalue = 0;

    for (int m = 0; m < Width/Tile_Width; ++m) {
        //joint loading
        subTileM[ty][tx] = M[Row*Width + m*TileWidth+tx];
        subTileN[ty][tx] = N[(m*Tile_Width+ty)*Width + Col];
        __synchthreads();
        for (int k=0; k < Tile_Width; ++k) {
            Pvalue += subTileM[ty][k] * subTileN[k][tx];
        __synchthreads();
        }
    P[Row*Width + Col] = Pvalue;
    }
}
