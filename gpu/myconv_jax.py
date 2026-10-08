import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler
import torch.utils.dlpack as tdl
import time
import json

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape

    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))
    _, _, h, w = x_pad.shape

    # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    # Refer to Lecture 3 for implementing this operation.

    k_i = jnp.arange(0, KH, 1)
    k_j = jnp.arange(0, KW, 1)
    h_i = jnp.arange(0, h - KH + 1, S)
    h_j = jnp.arange(0, w - KW + 1, S)
    h_i = h_i[:, None, None, None] + k_i[None, None, :, None]
    h_j = h_j[None, :, None, None] + k_j[None, None, None, :]

    patches = x_pad[:, :, h_i, h_j]
    patches = jnp.reshape(jnp.permute_dims(patches, (0, 2, 3, 1, 4, 5)), ((N, out_h * out_w, C * KH * KW)))

    
    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = (H + padding * 2 - (KH - 1) + stride - 1) // stride
    out_w = (W + padding * 2 - (KW - 1) + stride - 1) // stride
    
    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    weight = jnp.reshape(weight, (C_out, C * KH * KW))
    weight = jnp.permute_dims(weight, (1, 0))

    # TO DO: 3) perform tiled matmul after required reshaping is done.
    out = cols @ weight


    # TO DO: 4) Add bias.
    out = out + bias

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = jnp.permute_dims(out, (0, 2, 1))
    out = jnp.reshape(out, (N, C_out, out_h, out_w))

    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    
    with open("config.json") as config_file:
        config = json.load(config_file)
    
    N, C, H, W, K, S, P = config['N'], config['C'], config['H'], config['W'], config['K'], config['S'], config['P']


    model = ConvModel(H, W, in_channels=C, out_channels=C, kernel_size=K, stride=S, padding=P)
    model.eval()

    # Example input
    x_torch = torch.randn(1, C, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax, static_argnames="stride")
    lowered = conv2d_manual_jax_jit.lower(x_jax, weight_jax, bias_jax, stride = S)

    start_compile = time.perf_counter()
    compiled = lowered.compile()
    compile_time = time.perf_counter() - start_compile
    print(f"JAX Compilation Time: {compile_time * 1000:.2f} ms")

    # call your JAX function
    start = time.perf_counter()
    with jax.profiler.trace("./jax_trace", create_perfetto_trace=True):
        out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax, stride = S).block_until_ready()
    end = time.perf_counter()
    exe_time = end - start
    print(f"Execution Time: {exe_time * 1000:.2f} ms")

    # # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=S, padding=P)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(torch.from_numpy(np.array(out_jax)), conv_ref, atol=1e-1))
