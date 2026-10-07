// GPU test: the fused letterbox kernel must match the CPU reference
// (exactly for an unresized frame, within one uint8 step after resizing).
#include <cmath>
#include <random>
#include <vector>

#include "check.h"
#include "nurosim/cuda_utils.h"
#include "nurosim/letterbox.h"
#include "nurosim/preprocess_gpu.h"

using namespace nurosim;

static void compare(int w, int h, double max_tol, double min_exact_frac) {
  std::mt19937 rng(42);
  std::vector<uint8_t> frame(static_cast<size_t>(w) * h * 3);
  for (auto& v : frame) v = static_cast<uint8_t>(rng() & 0xFF);
  const auto m = compute_letterbox(w, h, 640);
  std::vector<float> ref(3 * 640 * 640), got(3 * 640 * 640);
  letterbox_cpu(frame.data(), m, ref.data());

  DeviceBuffer<uint8_t> d_frame(frame.size());
  DeviceBuffer<float> d_out(got.size());
  CUDA_CHECK(cudaMemcpy(d_frame.ptr, frame.data(), frame.size(), cudaMemcpyHostToDevice));
  letterbox_gpu(d_frame.ptr, m, d_out.ptr, nullptr);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(got.data(), d_out.ptr, got.size() * 4, cudaMemcpyDeviceToHost));

  double mx = 0;
  size_t exact = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = std::fabs(ref[i] - got[i]);
    mx = std::max(mx, d);
    exact += d == 0.0;
  }
  std::printf("  %dx%d: max |diff| %.6f, exact %.4f%%\n", w, h, mx, 100.0 * exact / ref.size());
  CHECK(mx <= max_tol);
  CHECK(static_cast<double>(exact) / ref.size() >= min_exact_frac);
}

TEST(identity_640_is_bit_exact) { compare(640, 640, 0.0, 1.0); }
TEST(resize_960x540_within_one_step) { compare(960, 540, 1.0 / 255 + 1e-6, 0.999); }
TEST(resize_1920x1080_within_one_step) { compare(1920, 1080, 1.0 / 255 + 1e-6, 0.999); }

int main() { return check::run_all(); }
