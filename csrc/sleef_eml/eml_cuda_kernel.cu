// Fused EML CUDA kernel: eml(x, y) = exp(x) - ln(y)
// Uses CUDA math intrinsics (expf/logf) which employ the same
// range-reduction + minimax polynomial approach as SLEEF.org.
// Forward + backward in fused single-kernel passes.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_exp(scalar_t x);

template <>
__device__ __forceinline__ float device_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ __forceinline__ double device_exp<double>(double x) {
    return exp(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t device_log(scalar_t x);

template <>
__device__ __forceinline__ float device_log<float>(float x) {
    return logf(x);
}

template <>
__device__ __forceinline__ double device_log<double>(double x) {
    return log(x);
}

template <typename scalar_t>
__global__ void eml_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ out,
    const int64_t n
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = device_exp(x[idx]) - device_log(y[idx]);
    }
}

template <typename scalar_t>
__global__ void eml_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ y,
    scalar_t* __restrict__ grad_x,
    scalar_t* __restrict__ grad_y,
    const int64_t n
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        scalar_t g = grad_output[idx];
        grad_x[idx] = g * device_exp(x[idx]);
        scalar_t yi = y[idx];
        if (yi < static_cast<scalar_t>(1e-30))
            yi = static_cast<scalar_t>(1e-30);
        grad_y[idx] = g * (static_cast<scalar_t>(-1.0) / yi);
    }
}

}  // anonymous namespace


torch::Tensor eml_cuda_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto out = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "eml_cuda_forward", ([&] {
        eml_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            n
        );
    }));

    return out;
}


std::vector<torch::Tensor> eml_cuda_backward(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor y
) {
    auto grad_x = torch::empty_like(x);
    auto grad_y = torch::empty_like(y);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "eml_cuda_backward", ([&] {
        eml_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            grad_x.data_ptr<scalar_t>(),
            grad_y.data_ptr<scalar_t>(),
            n
        );
    }));

    return {grad_x, grad_y};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &eml_cuda_forward, "Fused EML forward (CUDA)");
    m.def("backward", &eml_cuda_backward, "Fused EML backward (CUDA)");
}
