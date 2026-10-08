import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import time

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    N, C, H, W = 4, 16, 100, 100
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Torch-Inductor compilation
    scripted_model = torch.compile(model, backend="inductor")

    # (Compilation + First Execution)
    torch.cuda.synchronize()
    start_first = time.perf_counter()
    _ = scripted_model(x)
    torch.cuda.synchronize()
    first_run_time = time.perf_counter() - start_first

    # (No Compilation)
    torch.cuda.synchronize()
    start_steady = time.perf_counter()
    _ = scripted_model(x)
    torch.cuda.synchronize()
    steady_run_time = time.perf_counter() - start_steady

    # Host compilation time estimate
    compilation_time = first_run_time - steady_run_time
    print(f"Inductor Host Compilation Time: {compilation_time * 1000:.2f} ms")


    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = scripted_model(x)

    prof.export_chrome_trace("myconv_inductor.json")
    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))