// Host-side tests (no CUDA / TensorRT needed). Cases mirror tests/test_inference.py
// so the C++ and Python paths are held to the same expected values.
#include <cstdio>
#include <vector>

#include "check.h"
#include "nurosim/frames.h"
#include "nurosim/letterbox.h"
#include "nurosim/postprocess.h"
#include "nurosim/stats.h"

using namespace nurosim;

// [(cx, cy, w, h, cls, score)] → (4 + nc) x A row-major
static std::vector<float> raw(const std::vector<std::vector<float>>& anchors, int nc = 4) {
  const int A = static_cast<int>(anchors.size());
  std::vector<float> out((4 + nc) * A, 0.f);
  for (int a = 0; a < A; ++a) {
    for (int k = 0; k < 4; ++k) out[k * A + a] = anchors[a][k];
    out[(4 + static_cast<int>(anchors[a][4])) * A + a] = anchors[a][5];
  }
  return out;
}

TEST(square_frame_is_identity) {
  auto m = compute_letterbox(640, 640, 640);
  CHECK(m.ratio == 1.f && m.pad_x == 0.f && m.pad_y == 0.f && m.new_w == 640);
}

TEST(non_square_frame_is_padded) {
  auto m = compute_letterbox(960, 480, 640);
  CHECK_NEAR(m.ratio, 640.0 / 960.0, 1e-6);
  CHECK(m.pad_x == 0.f && m.pad_y == 160.f && m.new_h == 320);
}

TEST(letterbox_cpu_identity_and_layout) {
  std::vector<uint8_t> f(640 * 640 * 3);
  for (size_t i = 0; i < f.size(); i += 3) { f[i] = 255; f[i + 1] = 7; f[i + 2] = 0; }   // blue (BGR)
  auto m = compute_letterbox(640, 640, 640);
  std::vector<float> out(3 * 640 * 640);
  letterbox_cpu(f.data(), m, out.data());
  CHECK(out[0] == 0.f);                              // R plane
  CHECK(out[640 * 640] == 7.f * (1.f / 255.f));      // G plane, x * (1/255) like numpy
  CHECK(out[2 * 640 * 640] == 255.f * (1.f / 255.f));
}

TEST(letterbox_cpu_pads_with_114) {
  std::vector<uint8_t> f(480 * 960 * 3, 0);
  auto m = compute_letterbox(960, 480, 640);
  std::vector<float> out(3 * 640 * 640);
  letterbox_cpu(f.data(), m, out.data());
  CHECK(out[0] == 114.f * (1.f / 255.f));                   // top pad row
  CHECK(out[static_cast<size_t>(320) * 640 + 10] == 0.f);   // image region
}

TEST(boxes_mapped_back_through_letterbox) {
  auto m = compute_letterbox(960, 480, 640);
  auto r = raw({{320, 260, 60, 30, 0, 0.9f}});
  auto d = decode_yolov8(r.data(), 4, 1, m, 0.25f, 0.6f);
  CHECK(d.size() == 1);
  if (d.size() == 1) {
    CHECK(d[0].x1 == 435 && d[0].y1 == 128 && d[0].x2 == 525 && d[0].y2 == 172);
    CHECK(d[0].cls == 0);
    CHECK_NEAR(d[0].score, 0.9, 1e-6);
  }
}

TEST(confidence_threshold) {
  auto m = compute_letterbox(640, 640, 640);
  auto r = raw({{100, 100, 20, 20, 1, 0.9f}, {300, 300, 20, 20, 2, 0.1f}});
  auto d = decode_yolov8(r.data(), 4, 2, m, 0.25f, 0.6f);
  CHECK(d.size() == 1 && d[0].cls == 1);
}

TEST(nms_is_class_aware) {
  auto m = compute_letterbox(640, 640, 640);
  auto r = raw({{100, 100, 40, 40, 0, 0.9f}, {102, 101, 40, 40, 0, 0.8f}, {101, 100, 40, 40, 3, 0.7f}});
  auto d = decode_yolov8(r.data(), 4, 3, m, 0.25f, 0.6f);
  CHECK(d.size() == 2);
  if (d.size() == 2) CHECK(d[0].cls == 0 && d[1].cls == 3);
}

TEST(efficient_nms_mapping) {
  auto m = compute_letterbox(960, 480, 640);
  const float boxes[] = {290, 245, 350, 275, 0, 0, 0, 0};
  const float scores[] = {0.9f, 0.5f};
  const int32_t classes[] = {2, 0};
  auto d = from_efficient_nms(2, boxes, scores, classes, m);   // second box is degenerate
  CHECK(d.size() == 1);
  if (!d.empty()) CHECK(d[0].x1 == 435 && d[0].y1 == 128 && d[0].x2 == 525 && d[0].y2 == 172 && d[0].cls == 2);
}

TEST(percentile_matches_numpy) {
  std::vector<double> v = {1, 2, 3, 4, 10};
  CHECK_NEAR(percentile(v, 50), 3.0, 1e-12);
  CHECK_NEAR(percentile(v, 90), 7.6, 1e-12);       // np.percentile([1,2,3,4,10], 90)
  CHECK_NEAR(percentile(v, 99), 9.76, 1e-12);
  auto s = summarize(v);
  CHECK(s.n == 5 && s.max == 10);
  CHECK_NEAR(s.jitter_p99_p50, 6.76, 1e-12);
}

TEST(frames_roundtrip) {
  FrameSet fs;
  fs.n = 2; fs.h = 3; fs.w = 4; fs.c = 3;
  fs.data.resize(2 * 3 * 4 * 3);
  for (size_t i = 0; i < fs.data.size(); ++i) fs.data[i] = static_cast<uint8_t>(i);
  const char* p = "test_frames_roundtrip.bin";
  save_frames(p, fs);
  auto back = load_frames(p);
  std::remove(p);
  CHECK(back.n == 2 && back.h == 3 && back.w == 4 && back.c == 3 && back.data == fs.data);
  CHECK(back.frame(1)[0] == 36);
}

int main() { return check::run_all(); }
