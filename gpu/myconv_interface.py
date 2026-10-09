import torch
from torch.utils.cpp_extension import load
from torch.profiler import profile, ProfilerActivity
import time
import json

torch.backends.cudnn.allow_tf32 = False

# Compile and load CUDA extension
start_compile = time.perf_counter()
conv_module = load(name="myconv",
                     sources=["myconv_kernel.cu"],
                     verbose=True)

compile_time = time.perf_counter() - start_compile
print(f"CUDA Compilation Time: {compile_time:.2f} seconds")

# Input parameters

with open("config.json") as config_file:
    config = json.load(config_file)
    
    N, C, H, W, K, S, P = config['N'], config['C'], config['H'], config['W'], config['K'], config['S'], config['P']


C_in = C
C_out, KH, KW = C, K, K
stride, pad = S, P

# Allocate tensors
x = torch.randn(N, C_in, H, W, device="cuda", dtype=torch.float32)
w = torch.randn(C_out, C_in, KH, KW, device="cuda", dtype=torch.float32)

# Run o4 kernel
start = time.perf_counter()
with profile(activities=[ProfilerActivity.CUDA],) as prof:
    out_custom = conv_module.conv_cuda(x, w, stride, pad)
end = time.perf_counter()
exe_time = end - start
print(f"Execution Time: {exe_time * 1000:.2f} ms")

prof.export_chrome_trace("myconv_cuda.json")

# Reference solution (PyTorch)

out_ref = torch.nn.functional.conv2d(x, w, stride=stride, padding=pad)

diff = torch.abs(out_ref - out_custom)

# Test shape and correctness
print("CUDA --- shape check:", out_custom.shape == out_ref.shape)
print("CUDA --- correctness check:", torch.allclose(out_custom, out_ref, atol=1e-4, rtol= 1e-5))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
