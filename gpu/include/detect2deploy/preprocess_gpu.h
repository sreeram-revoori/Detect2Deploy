#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "detect2deploy/letterbox.h"

namespace detect2deploy {

// Fused letterbox on the GPU: bilinear resize + pad(114) + BGR→RGB + HWC→CHW +
// x*(1/255), one thread per output pixel, one launch per frame. Replaces the
// host-side Python/OpenCV preprocessing and shrinks the H2D copy 4x
// (uint8 HWC instead of float32 CHW).
void letterbox_gpu(const uint8_t* d_bgr, const LetterboxMeta& m, float* d_chw, cudaStream_t stream);

}  // namespace detect2deploy
