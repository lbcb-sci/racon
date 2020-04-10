// Implementation file for CUDA POA utilities.

#pragma once

#include <cuda_runtime_api.h>  // NOLINT

namespace racon {

void cudaCheckError(const std::string &msg) {
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "%s (CUDA error %s)\n", msg.c_str(), cudaGetErrorString(error));
    exit(-1);
  }
}

}  // namespace racon
