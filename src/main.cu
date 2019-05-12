#include <iostream>

inline void checkCuda(cudaError_t result, const char *file, const int line) {
    if (result != cudaSuccess) {
      std::cerr << file << "@" << line << ": CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
      exit(-1);
    }
  }

  #define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

int main(void) {
    std::cout << "enter main\n";

    cudaDeviceProp prop;
    CUDA_RUNTIME(cudaGetDeviceProperties(&prop, 0))

    std::cout << &prop << "\n";

    return 0;
}