#include "detect2deploy/postprocess.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace detect2deploy {

namespace {
constexpr float kMaxWH = 4096.f;   // class offset for batched NMS

// model-space xyxy → frame-space, clipped, rounded; false if degenerate
bool to_frame(float& x1, float& y1, float& x2, float& y2, const LetterboxMeta& m) {
  x1 = std::min(std::max((x1 - m.pad_x) / m.ratio, 0.f), static_cast<float>(m.frame_w));
  x2 = std::min(std::max((x2 - m.pad_x) / m.ratio, 0.f), static_cast<float>(m.frame_w));
  y1 = std::min(std::max((y1 - m.pad_y) / m.ratio, 0.f), static_cast<float>(m.frame_h));
  y2 = std::min(std::max((y2 - m.pad_y) / m.ratio, 0.f), static_cast<float>(m.frame_h));
  x1 = std::nearbyint(x1); y1 = std::nearbyint(y1);
  x2 = std::nearbyint(x2); y2 = std::nearbyint(y2);
  return x2 > x1 && y2 > y1;
}
}  // namespace

std::vector<int> nms(const std::vector<float>& b, const std::vector<float>& scores,
                     float iou_thr, int max_det) {
  const int n = static_cast<int>(scores.size());
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int i, int j) { return scores[i] > scores[j]; });
  std::vector<float> area(n);
  for (int i = 0; i < n; ++i)
    area[i] = std::max(0.f, b[4 * i + 2] - b[4 * i]) * std::max(0.f, b[4 * i + 3] - b[4 * i + 1]);

  std::vector<char> removed(n, 0);
  std::vector<int> keep;
  for (int oi = 0; oi < n && static_cast<int>(keep.size()) < max_det; ++oi) {
    const int i = order[oi];
    if (removed[i]) continue;
    keep.push_back(i);
    for (int oj = oi + 1; oj < n; ++oj) {
      const int j = order[oj];
      if (removed[j]) continue;
      const float iw = std::max(0.f, std::min(b[4 * i + 2], b[4 * j + 2]) - std::max(b[4 * i], b[4 * j]));
      const float ih = std::max(0.f, std::min(b[4 * i + 3], b[4 * j + 3]) - std::max(b[4 * i + 1], b[4 * j + 1]));
      const float inter = iw * ih, uni = area[i] + area[j] - inter;
      const float iou = uni > 0.f ? inter / uni : 0.f;
      if (iou > iou_thr) removed[j] = 1;
    }
  }
  return keep;
}

std::vector<Detection> decode_yolov8(const float* out, int nc, int A, const LetterboxMeta& m,
                                     float conf_thr, float iou_thr, int max_det) {
  std::vector<float> boxes, offset_boxes, scores;
  std::vector<int> classes;
  for (int a = 0; a < A; ++a) {
    int best = 0;
    float conf = out[4 * A + a];
    for (int c = 1; c < nc; ++c) {          // first max wins, like numpy argmax
      const float s = out[(4 + c) * A + a];
      if (s > conf) { conf = s; best = c; }
    }
    if (conf < conf_thr) continue;
    const float cx = out[a], cy = out[A + a], w = out[2 * A + a], h = out[3 * A + a];
    float x1 = cx - w / 2, y1 = cy - h / 2, x2 = cx + w / 2, y2 = cy + h / 2;
    x1 = std::min(std::max((x1 - m.pad_x) / m.ratio, 0.f), static_cast<float>(m.frame_w));
    x2 = std::min(std::max((x2 - m.pad_x) / m.ratio, 0.f), static_cast<float>(m.frame_w));
    y1 = std::min(std::max((y1 - m.pad_y) / m.ratio, 0.f), static_cast<float>(m.frame_h));
    y2 = std::min(std::max((y2 - m.pad_y) / m.ratio, 0.f), static_cast<float>(m.frame_h));
    boxes.insert(boxes.end(), {x1, y1, x2, y2});
    const float off = best * kMaxWH;
    offset_boxes.insert(offset_boxes.end(), {x1 + off, y1 + off, x2 + off, y2 + off});
    scores.push_back(conf);
    classes.push_back(best);
  }
  std::vector<Detection> dets;
  for (int k : nms(offset_boxes, scores, iou_thr, max_det)) {
    Detection d{std::nearbyint(boxes[4 * k]), std::nearbyint(boxes[4 * k + 1]),
                std::nearbyint(boxes[4 * k + 2]), std::nearbyint(boxes[4 * k + 3]),
                scores[k], classes[k]};
    if (d.x2 > d.x1 && d.y2 > d.y1) dets.push_back(d);
  }
  return dets;
}

std::vector<Detection> from_efficient_nms(int num, const float* boxes, const float* scores,
                                          const int32_t* classes, const LetterboxMeta& m) {
  std::vector<Detection> dets;
  for (int i = 0; i < num; ++i) {
    float x1 = boxes[4 * i], y1 = boxes[4 * i + 1], x2 = boxes[4 * i + 2], y2 = boxes[4 * i + 3];
    if (to_frame(x1, y1, x2, y2, m)) dets.push_back({x1, y1, x2, y2, scores[i], classes[i]});
  }
  return dets;
}

}  // namespace detect2deploy
