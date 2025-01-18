#include <cuda_runtime.h>

// CUDA kernel for element-wise multiplication
__global__ void elementwise_multiply_kernel(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for matrix multiplication
__global__ void matrix_multiply_kernel(const float* a, const float* b, float* c, 
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

// CUDA kernel for matrix transpose
__global__ void transpose_kernel(const float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

extern "C" {

// Wrapper function for element-wise multiplication
cudaError_t cuda_elementwise_multiply(const float* a, const float* b, float* c, int size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    elementwise_multiply_kernel<<<grid, block>>>(a, b, c, size);
    return cudaGetLastError();
}

// Wrapper function for matrix multiplication
cudaError_t cuda_matrix_multiply(const float* a, const float* b, float* c,
                               int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    matrix_multiply_kernel<<<grid, block>>>(a, b, c, M, N, K);
    return cudaGetLastError();
}

// Wrapper function for ReLU activation
cudaError_t cuda_relu(const float* input, float* output, int size) {
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    
    relu_kernel<<<grid, block>>>(input, output, size);
    return cudaGetLastError();
}

// Wrapper function for matrix transpose
cudaError_t cuda_transpose(const float* input, float* output, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    return cudaGetLastError();
}

} 