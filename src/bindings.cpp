#include <torch/extension.h>
#include "../cuda_ops/tensor_ops.cuh"

// Helper function to check CUDA errors
void check_cuda_error(cudaError_t error) {
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

// Python binding for element-wise multiplication
torch::Tensor elementwise_multiply(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same size");
    
    auto result = torch::empty_like(a);
    int size = a.numel();
    
    check_cuda_error(cuda_elementwise_multiply(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    ));
    
    return result;
}

// Python binding for matrix multiplication
torch::Tensor matrix_multiply(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "a and b must be 2D matrices");
    TORCH_CHECK(a.size(1) == b.size(0), "Invalid matrix dimensions for multiplication");
    
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    auto result = torch::empty({M, N}, a.options());
    
    check_cuda_error(cuda_matrix_multiply(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        M, N, K
    ));
    
    return result;
}

// Python binding for ReLU activation
torch::Tensor relu(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    
    auto result = torch::empty_like(input);
    int size = input.numel();
    
    check_cuda_error(cuda_relu(
        input.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    ));
    
    return result;
}

// Python binding for matrix transpose
torch::Tensor transpose(torch::Tensor input) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be a 2D matrix");
    
    int rows = input.size(0);
    int cols = input.size(1);
    
    auto result = torch::empty({cols, rows}, input.options());
    
    check_cuda_error(cuda_transpose(
        input.data_ptr<float>(),
        result.data_ptr<float>(),
        rows, cols
    ));
    
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_multiply", &elementwise_multiply, "Element-wise multiplication (CUDA)");
    m.def("matrix_multiply", &matrix_multiply, "Matrix multiplication (CUDA)");
    m.def("relu", &relu, "ReLU activation (CUDA)");
    m.def("transpose", &transpose, "Matrix transpose (CUDA)");
} 