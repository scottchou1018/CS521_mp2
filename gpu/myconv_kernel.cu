#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// example
#define TILE_H 16   
#define TILE_W 16   
#define TILE_C 16  


// Kernel declaration
__global__ void gemm_gpu_o4_kernel(
    const float* __restrict__ in,       // input: N x C x H x W
    const float* __restrict__ w,       // weights: C_out x C_in x KH x KW
    float* __restrict__ out,           // output: N x C x H x W
    int N, int C_in, int H, int W,
    int C_out, int KH, int KW,
    int stride, int pad,
    int out_h, int out_w
) {
    int bx = blockIdx.x, tx = threadIdx.x;
    int by = blockIdx.y, ty = threadIdx.y;
    int n = blockIdx.z / C_out;
    int i_out = blockIdx.z % C_out;
    extern __shared__ float shmem[];  // shared memory for partial sums
    // TO DO : Tiled matrix multiplication by using shmem
    int low_x = bx * blockDim.x * stride - pad;
    int low_y = by * blockDim.y * stride - pad;
    int input_h = 1 + (TILE_H - 1) * stride + KH - 1;
    int input_w = 1 + (TILE_W - 1) * stride + KW - 1;


    float *kernel = shmem; // TILE_C * KH * KW
    float *input = shmem + TILE_C * KH * KW; // TILE_C input_h * input_w
    float sum = 0;
    for(int c = 0; c < C_in; c += TILE_C){
        for(int j = 0; j < TILE_C && c + j < C_in; j++){
            // load kernel
            for(int x = 0; x < KH; x += TILE_H){
                for(int y = 0; y < KW; y += TILE_W){
                    if(x + tx < KH && y + ty < KW){
                        kernel[j * KH * KW + (x + tx) * KW + y + ty] = 
                        w[i_out * C_in * KH * KW + (c + j) * KH * KW + (x + tx) * KW + y + ty];
                    }
                }
            }
            // load input
            for(int x = 0; x < input_h; x += TILE_H){
                for(int y = 0; y < input_w; y += TILE_W){
                    if(x + tx < input_h && y + ty < input_w){
                        int x_id = x + low_x, y_id = y + low_y;
                        int tmp_id = j * input_h * input_w + (x + tx) * input_w + (y + ty);
                        if(x_id + tx >= 0 && x_id + tx < H && y_id + ty >= 0 && y_id + ty < W){
                            input[tmp_id] = in[n * C_in * H * W + (c + j) * H * W + (x_id + tx) * W + (y_id + ty)];
                        }else{
                            input[tmp_id] = 0;
                        }
                    }
                }
            }
        }
        __syncthreads();
        for(int j = 0; j < TILE_C && c + j < C_in; j++){
            for(int x = 0; x < KH; x++){
                for(int y = 0; y < KW; y++){
                    sum += kernel[j * KH * KW + x * KW + y] * 
                    input[j * input_h * input_w + (tx * stride + x) * input_w + (ty * stride + y)];
                }
            }
        }

        __syncthreads();
    }
    if(bx * TILE_H + tx < out_h && by * TILE_W + ty < out_w){
        out[n * C_out * out_h * out_w + i_out * out_h * out_w + (bx * TILE_H + tx) * out_w + (by * TILE_W + ty)] = sum;
    }

}

// Function for Python binding
torch::Tensor conv_cuda(torch::Tensor x, torch::Tensor w,
                          int stride, int pad) {
    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = w.size(0);
    int KH = w.size(2);
    int KW = w.size(3);

    int out_h = (H + pad * 2 - (KH - 1) + stride - 1) / stride;
    int out_w = (W + pad * 2 - (KW - 1) + stride - 1) / stride;

    auto out = torch::zeros({N, C_out, out_h, out_w}, x.options());

    dim3 block(TILE_H, TILE_W);
    dim3 grid((out_h + block.x - 1)/block.x,
              (out_w + block.y - 1)/block.y,
              N * C_out);
    
    int input_h = 1 + (TILE_H - 1) * stride + KH - 1;
    int input_w = 1 + (TILE_W - 1) * stride + KW - 1;
    size_t shmem_bytes = (TILE_C * KH * KW + TILE_C * input_h * input_w) * sizeof(float);
    gemm_gpu_o4_kernel<<<grid, block, shmem_bytes>>>(
        x.data_ptr<float>(),
        w.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C_in, H, W,
        C_out, KH, KW,
        stride, pad,
        out_h, out_w);

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv_cuda", &conv_cuda, "Custom Conv2D (CUDA)");
}
