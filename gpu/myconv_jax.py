import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler
import torch.utils.dlpack as tdl

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

    # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    # Refer to Lecture 3 for implementing this operation.

    patches = []
    _, _, h, w = x_pad.shape
    for i in range(0, h - KH + 1, S):
        for j in range(0, w - KW + 1, S):
            patches.append(jnp.reshape(x_pad[:, :, i: i + KH, j : j + KW], (N, C * KH * KW)))
    patches = jnp.stack(patches, 1)
    
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
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=2, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

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

    # call your JAX function
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax, stride = 2)
    out_jax = torch.tensor(np.array(out_jax))

    # # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=2, padding=1)
    print("JAX --- shape check:", out_jax.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(out_jax, conv_ref, atol=1e-1))
