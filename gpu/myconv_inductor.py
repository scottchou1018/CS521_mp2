import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import json
import time

torch.backends.cudnn.allow_tf32 = False

if __name__ == "__main__":
    torch.manual_seed(0)

    with open("config.json") as config_file:
        config = json.load(config_file)
    
    N, C, H, W, K, S, P = config['N'], config['C'], config['H'], config['W'], config['K'], config['S'], config['P']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Instantiate your PyTorch model
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=C, out_channels=C, kernel_size=K, stride=S, padding=P).cuda().eval()

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


    start = time.perf_counter()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        out = scripted_model(x)
    end = time.perf_counter()
    exe_time = end - start
    print(f"Execution Time: {exe_time * 1000:.2f} ms")

    prof.export_chrome_trace("myconv_inductor.json")
    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=S, padding=P)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-2))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))