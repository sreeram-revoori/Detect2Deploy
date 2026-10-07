#include "detect2deploy/frames.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace detect2deploy {

FrameSet load_frames(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[4];
  uint32_t hdr[4];
  f.read(magic, 4);
  f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
  if (!f || std::memcmp(magic, "NSFR", 4) != 0) throw std::runtime_error("bad frame file " + path);
  FrameSet fs;
  fs.n = static_cast<int>(hdr[0]);
  fs.h = static_cast<int>(hdr[1]);
  fs.w = static_cast<int>(hdr[2]);
  fs.c = static_cast<int>(hdr[3]);
  fs.data.resize(static_cast<size_t>(fs.n) * fs.frame_bytes());
  f.read(reinterpret_cast<char*>(fs.data.data()), static_cast<std::streamsize>(fs.data.size()));
  if (!f) throw std::runtime_error("truncated frame file " + path);
  return fs;
}

void save_frames(const std::string& path, const FrameSet& fs) {
  std::ofstream f(path, std::ios::binary);
  const uint32_t hdr[4] = {static_cast<uint32_t>(fs.n), static_cast<uint32_t>(fs.h),
                           static_cast<uint32_t>(fs.w), static_cast<uint32_t>(fs.c)};
  f.write("NSFR", 4);
  f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
  f.write(reinterpret_cast<const char*>(fs.data.data()), static_cast<std::streamsize>(fs.data.size()));
}

}  // namespace detect2deploy
