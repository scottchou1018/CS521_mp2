import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    # TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_H = 1
    TILE_W = 2
    assert TILE_H * TILE_W <= nl.tile_size.gemm_stationary_fmax
    # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
    # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_N = 128

    n_tiles_c_out = (out_channels + TILE_N - 1) // TILE_N

    hw = out_pool_height * out_pool_width
    # n_tiles_hw = (hw + TILE_M - 1) // TILE_M
    n_tile_h = (out_pool_height + TILE_H - 1) // TILE_H
    n_tile_w = (out_pool_width + TILE_W - 1) // TILE_W
    
    # Process the images in batches

    

    for b in nl.affine_range(batch_size):
        # nl.device_print("batch", b)
        for h_tile in nl.affine_range(n_tile_h):
            for w_tile in nl.affine_range(n_tile_w):
                for c_out_tile in nl.affine_range(n_tiles_c_out):
                    c_out_low = c_out_tile * TILE_N
                    c_out_high = c_out_low + TILE_N

                    h_low = h_tile * TILE_H
                    h_high = h_low + TILE_H
                    w_low = w_tile * TILE_W
                    w_high = w_low + TILE_W

                    out_psum = nl.ndarray(shape=(TILE_H * TILE_W, TILE_N),
                                        dtype=X_out.dtype,
                                        buffer=nl.psum)
                    for x in nl.affine_range(filter_height):
                        for y in nl.affine_range(filter_width):
                            for c_in_tile in nl.affine_range(n_tiles_c_in):
                                c_in_low = c_in_tile * c_in_pmax
                                c_in_high = (c_in_tile + 1) * c_in_pmax
                                X_t_sbuf = nl.ndarray(shape=(c_in_pmax, TILE_H, TILE_W),
                                                    dtype=X.dtype,
                                                    buffer=nl.sbuf)
                                W_sbuf = nl.ndarray(shape=(c_in_pmax, TILE_N),
                                                    dtype=W.dtype,
                                                    buffer=nl.sbuf)
                                # X_t_sbuf[...] = nl.load(X[b, ci, hi, wi])
                                X_t_sbuf[...] = nl.load(X[b, c_in_low:c_in_high, h_low + x : h_high + x, w_low + y:w_high + y])
                                
                                X_flat = X_t_sbuf.reshape((c_in_pmax, TILE_H * TILE_W))

                                load_W = W[c_out_low: c_out_high, c_in_low: c_in_high, x, y]
                                nisa.dma_copy(dst=W_sbuf, src=load_W)
                                W_T_sbuf = nisa.nc_transpose(W_sbuf)
                                out_psum += nl.matmul(X_flat[...], W_T_sbuf[...], transpose_x=True)

                    out_sbuf = nl.copy(out_psum, dtype=out_psum.dtype)
                    out_sbuf_T = nisa.nc_transpose(out_sbuf)


                    # bias_sbuf = nl.ndarray(TILE_N, dtype=bias.dtype, buffer=nl.sbuf)
                    bias_sbuf = nl.load(bias[c_out_low: c_out_high])
                    bias_sbuf_reshaped = bias_sbuf.reshape((TILE_N, 1))
                    out_bias = bias_sbuf + out_sbuf_T
                    out_sbuf_reshaped = out_bias.reshape((TILE_N, TILE_H, TILE_W))

                    nl.store(X_out[b, c_out_low : c_out_high, h_low: h_high, w_low : w_high],
                    value=out_sbuf_reshaped)




        # raise RuntimeError("Please fill your implementation of computing convolution"
        #                    " of X[b] with the weights W and bias b and store the result in X_out[b]")
    X_out = X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
    return X_out

