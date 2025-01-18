# CUDA Tensor Operations

A high-performance implementation of basic tensor operations using CUDA C, with Python bindings. This project demonstrates how to write custom CUDA kernels and compare their performance with PyTorch's native operations.

## Project Structure

```
GPU_ops/
├── src/
│   ├── cuda_ops/                   # CUDA Implementation
│   │   ├── tensor_ops.cu          # CUDA kernels
│   │   └── tensor_ops.cuh         # CUDA headers
│   └── matrix_ops.py          # High-level Python API
│   └── bindings.cpp           # PyTorch C++ bindings
├── tests/
│   └── test_tensor_ops.py         # Benchmarks & tests
├── setup.py                       # Build configuration
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation
```

## Features

### Implemented Operations
1. Matrix Multiplication
   - Optimized CUDA kernel for matrix multiplication
   - Support for arbitrary matrix dimensions
   - Efficient memory access patterns

2. Element-wise Multiplication
   - Parallel element-wise operations
   - Optimized for large tensors

3. ReLU Activation
   - Fast implementation of ReLU nonlinearity
   - Efficient in-place operation

4. Matrix Transpose
   - Cache-friendly matrix transpose
   - Optimized for different matrix sizes

### Performance Features
- Thread block optimization for NVIDIA GPUs
- Efficient memory access patterns
- Automatic synchronization handling
- PyTorch-compatible tensor operations

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested with v11.0+)
- Python 3.7+
- PyTorch with CUDA support
- Visual Studio Build Tools (Windows)

## Prerequisites

Before running this project, ensure that you have the following installed:

- **NVIDIA CUDA Toolkit**: This project requires the NVIDIA CUDA Toolkit to be installed on your system. You can download it from the [NVIDIA CUDA Toolkit Download page](https://developer.nvidia.com/cuda-downloads).

## Installation

1. Clone the repository.
2. Navigate to the project directory.
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Build the project:
   ```bash
   python setup.py build_ext --inplace
   ```

5. Run the tests:
   ```bash
   python -m pytest tests/test_tensor_ops.py
   ```

## Usage

### Basic Usage
```python
import torch
from src.python.matrix_ops import matrix_multiply, relu, transpose

# Create CUDA tensors
a = torch.randn(1000, 1000, device='cuda')
b = torch.randn(1000, 1000, device='cuda')

# Use our CUDA operations
c = matrix_multiply(a, b)           # Matrix multiplication
d = relu(c)                         # ReLU activation
e = transpose(d)                    # Matrix transpose
```

## Running Tests

### 1. Run All Tests
```bash
python -m tests.test_tensor_ops
```

### 2. Test Specific Operation
```bash
# Test matrix multiplication
python -m tests.test_tensor_ops --test multiply

# Test element-wise multiplication
python -m tests.test_tensor_ops --test elementwise

# Test ReLU activation
python -m tests.test_tensor_ops --test relu

# Test matrix transpose
python -m tests.test_tensor_ops --test transpose
```

### 3. Custom Matrix Size
```bash
python -m tests.test_tensor_ops --test multiply --size 2000
```

### Command-line Arguments
- `--test`: Operation to test
  - `multiply`: Matrix multiplication
  - `elementwise`: Element-wise multiplication
  - `relu`: ReLU activation
  - `transpose`: Matrix transpose
  - `all`: Run all tests (default)
- `--size`: Matrix size (N for NxN matrix, default: 1000)

## Implementation Details

### CUDA Kernels
- Optimized thread block sizes (16x16 for 2D operations)
- Efficient memory coalescing
- Automatic grid size calculation
- Error checking and validation

### Python Integration
- PyTorch tensor compatibility
- Automatic memory management
- Type checking and validation
- Performance benchmarking

### Performance Considerations
- Thread block size optimization
- Memory access patterns
- Synchronization handling
- Error checking overhead

## Troubleshooting

### Common Issues
1. CUDA_HOME not set:
   - Verify CUDA Toolkit installation
   - Set environment variable correctly

2. Compiler errors (Windows):
   - Use Developer Command Prompt
   - Verify VS Build Tools installation
   - Check Windows SDK installation

3. Build failures:
   - Check CUDA installation
   - Verify PATH includes CUDA
   - Ensure correct compiler setup 