import numpy as np
import torch
import time
import argparse
from src.matrix_ops import matrix_multiply, elementwise_multiply, relu, transpose

def create_test_matrices(M, N):
    """Create random test matrices."""
    A = np.random.rand(M, N).astype(np.float32)
    B = np.random.rand(M, N).astype(np.float32)
    return torch.from_numpy(A).cuda(), torch.from_numpy(B).cuda()

def benchmark_operation(operation_name, our_op, torch_op, *inputs):
    """Generic benchmarking function for operations."""
    print(f"\n{operation_name}:")
    
    # Our implementation
    torch.cuda.synchronize()
    start_time = time.time()
    our_result = our_op(*inputs)
    torch.cuda.synchronize()
    our_time = time.time() - start_time
    
    # PyTorch implementation
    torch.cuda.synchronize()
    start_time = time.time()
    torch_result = torch_op(*inputs)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time
    
    # Compare results
    max_diff = torch.max(torch.abs(our_result - torch_result)).item()
    print(f"Our implementation time: {our_time:.4f} seconds")
    print(f"PyTorch time: {torch_time:.4f} seconds")
    print(f"Maximum difference: {max_diff:.8f}")
    
    return our_time, torch_time, max_diff

def test_matrix_multiply(size):
    """Test matrix multiplication operation."""
    M, N = size
    print(f"\nMatrix size: {M}x{N}")
    A, B = create_test_matrices(M, N)
    
    return benchmark_operation(
        "Matrix Multiplication",
        matrix_multiply,
        torch.matmul,
        A, B
    )

def test_elementwise_multiply(size):
    """Test element-wise multiplication operation."""
    M, N = size
    A, B = create_test_matrices(M, N)
    
    return benchmark_operation(
        "Element-wise Multiplication",
        elementwise_multiply,
        lambda x, y: x * y,
        A, B
    )

def test_relu(size):
    """Test ReLU activation operation."""
    M, N = size
    A, _ = create_test_matrices(M, N)
    
    return benchmark_operation(
        "ReLU Activation",
        relu,
        torch.relu,
        A
    )

def test_transpose(size):
    """Test matrix transpose operation."""
    M, N = size
    A, _ = create_test_matrices(M, N)
    
    return benchmark_operation(
        "Matrix Transpose",
        transpose,
        lambda x: x.t(),
        A
    )

def run_all_tests(sizes=None):
    """Run all tensor operation tests."""
    if sizes is None:
        sizes = [(1000, 1000), (2000, 2000), (4000, 4000)]
    
    print("Matrix Operations Benchmark")
    print("==========================")
    
    results = {}
    for size in sizes:
        print(f"\nTesting size: {size[0]}x{size[1]}")
        results[size] = {
            'matrix_multiply': test_matrix_multiply(size),
            'elementwise_multiply': test_elementwise_multiply(size),
            'relu': test_relu(size),
            'transpose': test_transpose(size)
        }
    
    # Print summary
    print("\nTest Summary")
    print("============")
    for size, ops in results.items():
        print(f"\nSize {size[0]}x{size[1]}:")
        for op_name, (our_time, torch_time, max_diff) in ops.items():
            print(f"\n{op_name}:")
            print(f"  Our implementation: {our_time:.4f}s")
            print(f"  PyTorch: {torch_time:.4f}s")
            print(f"  Speed ratio: {torch_time/our_time:.2f}x")
            print(f"  Max difference: {max_diff:.8f}")

def run_specific_test(test_name, size):
    """Run a specific test with given size."""
    test_functions = {
        'multiply': test_matrix_multiply,
        'elementwise': test_elementwise_multiply,
        'relu': test_relu,
        'transpose': test_transpose
    }
    
    if test_name not in test_functions:
        print(f"Error: Unknown test '{test_name}'")
        print(f"Available tests: {list(test_functions.keys())}")
        return
    
    test_func = test_functions[test_name]
    our_time, torch_time, max_diff = test_func(size)
    
    print(f"\nSummary for {test_name}:")
    print(f"  Our implementation: {our_time:.4f}s")
    print(f"  PyTorch: {torch_time:.4f}s")
    print(f"  Speed ratio: {torch_time/our_time:.2f}x")
    print(f"  Max difference: {max_diff:.8f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tensor operation tests')
    parser.add_argument('--test', type=str, choices=['multiply', 'elementwise', 'relu', 'transpose', 'all'],
                      help='Specific test to run')
    parser.add_argument('--size', type=int, default=1000,
                      help='Matrix size (N for NxN matrix)')
    
    args = parser.parse_args()
    
    if args.test and args.test != 'all':
        run_specific_test(args.test, (args.size, args.size))
    else:
        sizes = [(args.size, args.size)] if args.size != 1000 else None
        run_all_tests(sizes) 