import torch
from src.gpu_ops._cuda import matrix_multiply as _matrix_multiply
from src.gpu_ops._cuda import elementwise_multiply as _elementwise_multiply
from src.gpu_ops._cuda import relu as _relu
from src.gpu_ops._cuda import transpose as _transpose

def matrix_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication using our CUDA implementation."""
    return _matrix_multiply(a, b)

def elementwise_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise multiplication using our CUDA implementation."""
    return _elementwise_multiply(a, b)

def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU activation using our CUDA implementation."""
    return _relu(x)

def transpose(x: torch.Tensor) -> torch.Tensor:
    """Matrix transpose using our CUDA implementation."""
    return _transpose(x) 