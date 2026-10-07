#pragma once
// Frame sets dumped by `python -m detect2deploy.deploy trt-prep`:
//   "NSFR" | uint32 n | uint32 h | uint32 w | uint32 c | n*h*w*c bytes (BGR uint8)
// Keeps the C++ tools free of an image-decoding dependency.
#include <cstdint>
#include <string>
#include <vector>

namespace detect2deploy {

struct FrameSet {
  int n = 0, h = 0, w = 0, c = 0;
  std::vector<uint8_t> data;
  size_t frame_bytes() const { return static_cast<size_t>(h) * w * c; }
  const uint8_t* frame(int i) const { return data.data() + static_cast<size_t>(i % n) * frame_bytes(); }
};

FrameSet load_frames(const std::string& path);   // throws std::runtime_error
void save_frames(const std::string& path, const FrameSet& fs);

}  // namespace detect2deploy
