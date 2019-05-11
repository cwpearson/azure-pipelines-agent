#include <iostream>



int main(void) {
    std::cout << "launched main\n";

    cudaDeviceProp prop;
    cudaError_t ret = cudaGetDeviceProperties(&prop, 0);
    if (ret != cudaSuccess) {
        std::cout << "error\n";
        return 1;
    }

    std::cout << &prop << "\n";

    return 0;
}