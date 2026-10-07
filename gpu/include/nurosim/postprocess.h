#pragma once
// YOLOv8 head decode + class-aware NMS (CPU), and mapping of TensorRT
// EfficientNMS outputs back to frame space. Same semantics as
// nurosim/inference/postprocess.py: boxes are rounded to whole pixels and
// degenerate boxes are dropped, so outputs compare 1:1 with the Python path.
#include <cstdint>
#include <vector>

#include "nurosim/letterbox.h"

namespace nurosim {

struct Detection {
  float x1, y1, x2, y2;
  float score;
  int cls;
};

// Greedy NMS over xyxy boxes (flattened, 4 per box). Returns kept indices.
std::vector<int> nms(const std::vector<float>& boxes, const std::vector<float>& scores,
                     float iou_thr, int max_det);

// out: one image's raw head output, row-major (4 + num_classes) x num_anchors.
std::vector<Detection> decode_yolov8(const float* out, int num_classes, int num_anchors,
                                     const LetterboxMeta& m, float conf_thr, float iou_thr,
                                     int max_det = 300);

// EfficientNMS_TRT outputs for one image: boxes are xyxy in model space.
std::vector<Detection> from_efficient_nms(int num_dets, const float* boxes, const float* scores,
                                          const int32_t* classes, const LetterboxMeta& m);

}  // namespace nurosim
