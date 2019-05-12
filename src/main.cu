#include <iostream>

inline void checkCuda(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
        std::cerr << file << "@" << line << ": CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
        exit(-1);
    }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

template <typename T>
__global__ void setter_kernel(T *x, const size_t n) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] = 10;
    }
}

int main(void) {
    std::cout << "enter main\n";

    cudaDeviceProp prop;
    CUDA_RUNTIME(cudaGetDeviceProperties(&prop, 0))

    std::cout << prop.name << "\n";

    float *x;
    CUDA_RUNTIME(cudaMallocManaged(&x, sizeof(*x) * 10));
    setter_kernel<<<10, 512>>>(x, 10);
    CUDA_RUNTIME(cudaDeviceSynchronize());

    if (x[0] != 10) {
        std::cerr << "setter kernel failed\n";
        return 1;
    }

    CUDA_RUNTIME(cudaFree(x));

    std::cerr << "tests passed\n";
    return 0;
}