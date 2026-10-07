#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define D2D_HAS_NVTX 1
#endif

#define CUDA_CHECK(expr)                                                              \
  do {                                                                                \
    cudaError_t e_ = (expr);                                                          \
    if (e_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, \
                   cudaGetErrorString(e_));                                           \
      std::abort();                                                                   \
    }                                                                                 \
  } while (0)

namespace detect2deploy {

// NVTX range so stages show up by name in Nsight Systems timelines.
struct NvtxRange {
  explicit NvtxRange(const char* name) {
#ifdef D2D_HAS_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
  }
  ~NvtxRange() {
#ifdef D2D_HAS_NVTX
    nvtxRangePop();
#endif
  }
};

// Pinned (page-locked) host buffer: required for truly async H2D/D2H copies.
template <typename T>
struct PinnedBuffer {
  T* ptr = nullptr;
  size_t count = 0;
  explicit PinnedBuffer(size_t n) : count(n) { CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T))); }
  ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

template <typename T>
struct DeviceBuffer {
  T* ptr = nullptr;
  size_t count = 0;
  explicit DeviceBuffer(size_t n) : count(n) { CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T))); }
  ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

}  // namespace detect2deploy
