#ifndef TENSOR_OPS_CUH
#define TENSOR_OPS_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise multiplication of two tensors
cudaError_t cuda_elementwise_multiply(const float* a, const float* b, float* c, int size);

// Matrix multiplication (C = A * B)
cudaError_t cuda_matrix_multiply(const float* a, const float* b, float* c,
                               int M, int N, int K);

// ReLU activation function
cudaError_t cuda_relu(const float* input, float* output, int size);

// Matrix transpose
cudaError_t cuda_transpose(const float* input, float* output, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_OPS_CUH 