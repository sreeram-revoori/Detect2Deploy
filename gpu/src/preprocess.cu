#include "detect2deploy/preprocess_gpu.h"

namespace detect2deploy {

namespace {

__global__ void letterbox_kernel(const uint8_t* __restrict__ src, int W, int H, int new_w, int new_h,
                                 int left, int top, float sx, float sy, bool identity, int S,
                                 float* __restrict__ dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= S || y >= S) return;

  const int rx = x - left, ry = y - top;
  float b, g, r;
  if (rx < 0 || ry < 0 || rx >= new_w || ry >= new_h) {
    b = g = r = 114.f;
  } else if (identity) {
    const uint8_t* p = src + (static_cast<size_t>(ry) * W + rx) * 3;
    b = p[0]; g = p[1]; r = p[2];
  } else {
    // Same sampling as letterbox_cpu (cv2 INTER_LINEAR, half-pixel centres)
    float fx = (rx + 0.5f) * sx - 0.5f, fy = (ry + 0.5f) * sy - 0.5f;
    int x0 = static_cast<int>(floorf(fx)), y0 = static_cast<int>(floorf(fy));
    fx -= x0; fy -= y0;
    if (x0 < 0) { x0 = 0; fx = 0.f; }
    if (y0 < 0) { y0 = 0; fy = 0.f; }
    if (x0 >= W - 1) { x0 = W - 1; fx = 0.f; }
    if (y0 >= H - 1) { y0 = H - 1; fy = 0.f; }
    const int x1 = min(x0 + 1, W - 1), y1 = min(y0 + 1, H - 1);
    const uint8_t* p00 = src + (static_cast<size_t>(y0) * W + x0) * 3;
    const uint8_t* p01 = src + (static_cast<size_t>(y0) * W + x1) * 3;
    const uint8_t* p10 = src + (static_cast<size_t>(y1) * W + x0) * 3;
    const uint8_t* p11 = src + (static_cast<size_t>(y1) * W + x1) * 3;
    float v[3];
    for (int c = 0; c < 3; ++c) {
      const float t = (p00[c] * (1 - fx) + p01[c] * fx) * (1 - fy) + (p10[c] * (1 - fx) + p11[c] * fx) * fy;
      v[c] = fminf(255.f, fmaxf(0.f, rintf(t)));
    }
    b = v[0]; g = v[1]; r = v[2];
  }
  const float inv255 = 1.0f / 255.0f;
  const size_t plane = static_cast<size_t>(S) * S, i = static_cast<size_t>(y) * S + x;
  dst[i] = r * inv255;
  dst[plane + i] = g * inv255;
  dst[2 * plane + i] = b * inv255;
}

}  // namespace

void letterbox_gpu(const uint8_t* d_bgr, const LetterboxMeta& m, float* d_chw, cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid((m.size + block.x - 1) / block.x, (m.size + block.y - 1) / block.y);
  const bool identity = (m.new_w == m.frame_w && m.new_h == m.frame_h);
  letterbox_kernel<<<grid, block, 0, stream>>>(
      d_bgr, m.frame_w, m.frame_h, m.new_w, m.new_h, static_cast<int>(m.pad_x),
      static_cast<int>(m.pad_y), static_cast<float>(m.frame_w) / m.new_w,
      static_cast<float>(m.frame_h) / m.new_h, identity, m.size, d_chw);
}

}  // namespace detect2deploy
