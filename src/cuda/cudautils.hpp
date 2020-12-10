// Implementation file for CUDA POA utilities.

#ifndef RACON_CUDA_CUDAUTILS_HPP_
#define RACON_CUDA_CUDAUTILS_HPP_

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>

namespace racon {

void CudaCheckError(const std::string &msg) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << msg << " (CUDA error " << cudaGetErrorString(error) << ")"
              << std::endl;
    exit(-1);
  }
}

}  // namespace racon

#endif  // RACON_CUDA_CUDAUTILS_HPP_
