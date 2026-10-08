import torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity
import time

# Compile and load CUDA extension
start_compile = time.perf_counter()
conv_module = load(name="myconv",
                     sources=["myconv_kernel.cu"],
                     verbose=True)

compile_time = time.perf_counter() - start_compile
print(f"CUDA Compilation Time: {compile_time:.2f} seconds")

# Input parameters
N, C_in, H, W = 4, 3, 30, 25
C_out, KH, KW = 4, 5, 5
stride, pad = 2, 3

# Allocate tensors
x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
w = torch.randn(C_out, C_in, KH, KW, device="cuda", dtype=torch.float32)

# Run o4 kernel
with profile(activities=[ProfilerActivity.CUDA],) as prof:
    out_custom = conv_module.conv_cuda(x, w, stride, pad)

# Reference solution (PyTorch)
out_ref = torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)

# Test shape and correctness
print("CUDA --- shape check:", out_custom.shape == out_ref.shape)
print("CUDA --- correctness check:", torch.allclose(out_custom, out_ref, atol=1e-4))
