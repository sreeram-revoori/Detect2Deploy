#include "detect2deploy/letterbox.h"

#include <algorithm>
#include <cmath>

namespace detect2deploy {

LetterboxMeta compute_letterbox(int frame_w, int frame_h, int size) {
  LetterboxMeta m{};
  const double r = std::min(static_cast<double>(size) / frame_h,
                            static_cast<double>(size) / frame_w);
  // std::nearbyint uses round-half-even, matching Python's round()
  m.new_w = static_cast<int>(std::nearbyint(frame_w * r));
  m.new_h = static_cast<int>(std::nearbyint(frame_h * r));
  const double pad_x = (size - m.new_w) / 2.0, pad_y = (size - m.new_h) / 2.0;
  m.pad_x = static_cast<float>(std::nearbyint(pad_x - 0.1));
  m.pad_y = static_cast<float>(std::nearbyint(pad_y - 0.1));
  m.ratio = static_cast<float>(r);
  m.frame_w = frame_w;
  m.frame_h = frame_h;
  m.size = size;
  return m;
}

void letterbox_cpu(const uint8_t* bgr, const LetterboxMeta& m, float* out) {
  const int S = m.size, W = m.frame_w, H = m.frame_h;
  const int left = static_cast<int>(m.pad_x), top = static_cast<int>(m.pad_y);
  const float sx = static_cast<float>(W) / m.new_w, sy = static_cast<float>(H) / m.new_h;
  const bool identity = (m.new_w == W && m.new_h == H);
  const float inv255 = 1.0f / 255.0f;
  const size_t plane = static_cast<size_t>(S) * S;

  for (int y = 0; y < S; ++y) {
    for (int x = 0; x < S; ++x) {
      const int rx = x - left, ry = y - top;
      float px[3];
      if (rx < 0 || ry < 0 || rx >= m.new_w || ry >= m.new_h) {
        px[0] = px[1] = px[2] = kPadValue;
      } else if (identity) {
        const uint8_t* p = bgr + (static_cast<size_t>(ry) * W + rx) * 3;
        px[0] = p[0]; px[1] = p[1]; px[2] = p[2];
      } else {
        // cv2 INTER_LINEAR: half-pixel centres, clamp at the borders
        float fx = (rx + 0.5f) * sx - 0.5f, fy = (ry + 0.5f) * sy - 0.5f;
        int x0 = static_cast<int>(std::floor(fx)), y0 = static_cast<int>(std::floor(fy));
        fx -= x0; fy -= y0;
        if (x0 < 0) { x0 = 0; fx = 0.f; }
        if (y0 < 0) { y0 = 0; fy = 0.f; }
        if (x0 >= W - 1) { x0 = W - 1; fx = 0.f; }
        if (y0 >= H - 1) { y0 = H - 1; fy = 0.f; }
        const int x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
        for (int c = 0; c < 3; ++c) {
          const float a = bgr[(static_cast<size_t>(y0) * W + x0) * 3 + c];
          const float b = bgr[(static_cast<size_t>(y0) * W + x1) * 3 + c];
          const float d = bgr[(static_cast<size_t>(y1) * W + x0) * 3 + c];
          const float e = bgr[(static_cast<size_t>(y1) * W + x1) * 3 + c];
          const float v = (a * (1 - fx) + b * fx) * (1 - fy) + (d * (1 - fx) + e * fx) * fy;
          px[c] = std::min(255.f, std::max(0.f, std::nearbyint(v)));   // resize output is uint8
        }
      }
      const size_t i = static_cast<size_t>(y) * S + x;
      out[i]             = px[2] * inv255;   // R
      out[plane + i]     = px[1] * inv255;   // G
      out[2 * plane + i] = px[0] * inv255;   // B
    }
  }
}

}  // namespace detect2deploy
