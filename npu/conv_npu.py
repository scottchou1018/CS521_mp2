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
    TILE_H = 2
    TILE_W = out_width
    assert TILE_H * TILE_W <= nl.tile_size.gemm_moving_fmax
    # Maximum free dimension of the moving operand of general matrix multiplication on tensor engine
    # TILE_N = nl.tile_size.gemm_moving_fmax  # 512
    TILE_N = nl.tile_size.gemm_stationary_fmax # 128

    n_tiles_c_out = out_channels // TILE_N

    hw = out_pool_height * out_pool_width
    n_tile_h = out_pool_height // TILE_H
    n_tile_w = out_pool_width // TILE_W


    remind_h = out_pool_height - n_tile_h * TILE_H
    remind_w = out_pool_width - n_tile_w * TILE_W

    W_sbuf = nl.ndarray(shape=(c_in_pmax, n_tiles_c_out, n_tiles_c_in, filter_height, filter_width, TILE_N),
                        dtype=W.dtype,
                        buffer=nl.sbuf)
    
    
    for c_out_tile in nl.affine_range(n_tiles_c_out):
        for c_in_tile in nl.affine_range(n_tiles_c_in):
            for x in nl.affine_range(filter_height):
                for y in nl.affine_range(filter_width):
                    # W_sbuf[:,c_out_tile,c_in_tile,x,y,:] = nl.load(W[c_out_tile * TILE_N: (c_out_tile + 1) * TILE_N, c_in_tile * c_in_pmax : (c_in_tile + 1) * c_in_pmax, x, y])
                    W_sbuf_tmp = nl.ndarray(shape=(c_in_pmax, TILE_N),
                                dtype=W.dtype,
                                buffer=nl.sbuf)
                    nisa.dma_copy(dst = W_sbuf_tmp, src=W[c_out_tile * TILE_N: (c_out_tile + 1) * TILE_N, c_in_tile * c_in_pmax : (c_in_tile + 1) * c_in_pmax, x, y])
                    W_sbuf[:,c_out_tile,c_in_tile,x,y,:] = nisa.nc_transpose(W_sbuf_tmp)

                    # nisa.dma_copy(dst = W_sbuf[:,c_out_tile,c_in_tile,x,y,:], src=W[c_out_tile * TILE_N: (c_out_tile + 1) * TILE_N, c_in_tile * c_in_pmax : (c_in_tile + 1) * c_in_pmax, x, y])

    
    for b in nl.affine_range(batch_size):
        # for h_tile in nl.affine_range(n_tile_h):
        for h_id in nl.affine_range(out_pool_height):

            X_t_sbuf = nl.ndarray(shape=(c_in_pmax, n_tiles_c_in, filter_height, input_width),
                                  dtype=X.dtype,
                                  buffer=nl.sbuf)
            for c_in_tile in nl.affine_range(n_tiles_c_in):
                c_in_low = c_in_tile * c_in_pmax
                c_in_high = (c_in_tile + 1) * c_in_pmax
                
                for x in nl.affine_range(filter_height):
                    X_t_sbuf[:, c_in_tile, x, :] = nl.load(X[b, c_in_low : c_in_high, x + h_id, :])

            for c_out_tile in nl.affine_range(n_tiles_c_out):
                c_out_low = c_out_tile * TILE_N
                c_out_high = c_out_low + TILE_N

                out_psum = nl.ndarray(shape=(TILE_N, out_pool_width),
                                    dtype=X_out.dtype,
                                    buffer=nl.psum)
                for c_in_tile in nl.affine_range(n_tiles_c_in):
                    c_in_low = c_in_tile * c_in_pmax
                    c_in_high = (c_in_tile + 1) * c_in_pmax
                    
                    for x in nl.affine_range(filter_height):
                        # X_t_sbuf = nl.ndarray(shape=(c_in_pmax, input_width),
                        #                     dtype=X.dtype,
                        #                     buffer=nl.sbuf)
                        # X_t_sbuf = nl.load(X[b, c_in_low : c_in_high, x + h_id, :])

                        for y in nl.affine_range(filter_width):

                            # X_t_sbuf = nl.ndarray(shape=(c_in_pmax, out_pool_width),
                            #                     dtype=X.dtype,
                            #                     buffer=nl.sbuf)
                            # X_t_sbuf = nl.load(X[b, c_in_low : c_in_high, x + h_id, y: y + out_pool_width])

                            # W_T_sbuf = nisa.nc_transpose(W_sbuf[:,c_out_tile,c_in_tile,x,y,:])
                            out_psum += nl.matmul(W_sbuf[:,c_out_tile, c_in_tile,x,y,:], X_t_sbuf[:, c_in_tile, x,y : y + out_pool_width], transpose_x=True)

                out_sbuf = nl.copy(out_psum, dtype=out_psum.dtype)

                # bias_sbuf = nl.ndarray(TILE_N, dtype=bias.dtype, buffer=nl.sbuf)
                bias_sbuf = nl.load(bias[c_out_low : c_out_high])
                # bias_sbuf_reshaped = bias_sbuf.reshape((TILE_N, 1))
                out_bias = bias_sbuf + out_sbuf

                nl.store(X_out[b, c_out_tile * TILE_N: (c_out_tile + 1) * TILE_N, h_id, :],
                value=out_bias)
    


    X_out = X_out.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
    return X_out

