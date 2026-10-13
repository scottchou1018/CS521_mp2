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
    TILE_H = 16
    TILE_W = 32
    assert TILE_H * TILE_W <= nl.tile_size.gemm_moving_fmax
    # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
    # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_N = nl.tile_size.gemm_stationary_fmax # 128

    n_tiles_c_out = out_channels // TILE_N

    hw = out_pool_height * out_pool_width
    # n_tiles_hw = (hw + TILE_M - 1) // TILE_M
    n_tile_h = out_pool_height // TILE_H
    n_tile_w = out_pool_width // TILE_W
    
    # Process the images in batches

    remind_h = out_pool_height - n_tile_h * TILE_H
    remind_w = out_pool_width - n_tile_w * TILE_W

    def compute(h_tile, h_size, w_tile, w_size):
        for c_out_tile in nl.affine_range(n_tiles_c_out):
            c_out_low = c_out_tile * TILE_N
            c_out_high = c_out_low + TILE_N

            h_low = h_tile * TILE_H
            h_high = h_low + h_size
            w_low = w_tile * TILE_W
            w_high = w_low + w_size

            out_psum = nl.ndarray(shape=(TILE_N, h_size * w_size),
                                dtype=X_out.dtype,
                                buffer=nl.psum)
            for x in nl.affine_range(filter_height):
                hi = x + h_low + nl.arange(h_size)[None, :, None]
                for y in nl.affine_range(filter_width):
                    wi = y + w_low + nl.arange(w_size)[None, None, :]
                    for c_in_tile in nl.affine_range(n_tiles_c_in):
                        c_in_low = c_in_tile * c_in_pmax
                        c_in_high = (c_in_tile + 1) * c_in_pmax
                        X_t_sbuf = nl.ndarray(shape=(c_in_pmax, h_size, w_size),
                                            dtype=X.dtype,
                                            buffer=nl.sbuf)
                        W_sbuf = nl.ndarray(shape=(c_in_pmax, TILE_N),
                                            dtype=W.dtype,
                                            buffer=nl.sbuf)
                        # X_t_sbuf[...] = nl.load(X[b, ci, hi, wi])
                        # X_t_sbuf[...] = nl.load(X[b, c_in_low:c_in_high, h_low + x : h_high + x, w_low + y:w_high + y])
                        ci = c_in_low + nl.arange(c_in_pmax)[:, None, None]

                        nisa.dma_copy(dst=X_t_sbuf, src=X[b, ci, hi, wi])

                        X_flat = X_t_sbuf.reshape((c_in_pmax, h_size * w_size))

                        load_W = W[c_out_low: c_out_high, c_in_low: c_in_high, x, y]
                        nisa.dma_copy(dst=W_sbuf, src=load_W)
                        W_T_sbuf = nisa.nc_transpose(W_sbuf)
                        out_psum += nl.matmul(W_T_sbuf[...], X_flat[...], transpose_x=True)

            out_sbuf = nl.copy(out_psum, dtype=out_psum.dtype)

            # bias_sbuf = nl.ndarray(TILE_N, dtype=bias.dtype, buffer=nl.sbuf)
            bias_sbuf = nl.load(bias[c_out_low: c_out_high])
            # bias_sbuf_reshaped = bias_sbuf.reshape((TILE_N, 1))
            out_bias = bias_sbuf + out_sbuf
            out_sbuf_reshaped = out_bias.reshape((TILE_N, h_size, w_size))

            nl.store(X_out[b, c_out_low : c_out_high, h_low: h_high, w_low : w_high],
            value=out_sbuf_reshaped)

    for b in nl.affine_range(batch_size):
        # nl.device_print("batch", b)
        for h_tile in nl.affine_range(n_tile_h):
            # h_low = h_tile * TILE_H
            # h_high = h_low + TILE_H
            for w_tile in nl.affine_range(n_tile_w):
                compute(h_tile, TILE_H, w_tile, TILE_W)


    if remind_w > 0:
        for b in nl.affine_range(batch_size):
                # nl.device_print("batch", b)
                for h_tile in nl.affine_range(n_tile_h):
                    compute(h_tile, TILE_H, n_tile_w, remind_w)

    if remind_h > 0:
        for b in nl.affine_range(batch_size):
            # nl.device_print("batch", b)
            for w_tile in nl.affine_range(n_tile_w):
                compute(n_tile_h, remind_h, w_tile, TILE_W)

    if remind_h > 0 and remind_w > 0:
            for b in nl.affine_range(batch_size):
                compute(n_tile_h, remind_h, n_tile_w, remind_w)
    


    X_out = X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
    return X_out

