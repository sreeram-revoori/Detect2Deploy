#pragma once
// Letterbox geometry + CPU reference preprocessing.
// Mirrors nurosim/inference/preprocess.py exactly (Python's round-half-even,
// cv2 INTER_LINEAR half-pixel sampling, pad value 114, x * (1/255) scaling) so
// C++ and Python detections can be compared box-for-box.
#include <cstdint>

namespace nurosim {

constexpr int kPadValue = 114;

struct LetterboxMeta {
  float ratio;    // model px / frame px
  float pad_x;    // left padding (model px)
  float pad_y;    // top padding (model px)
  int frame_w, frame_h;
  int new_w, new_h;   // resized frame size inside the letterbox
  int size;           // square model input size
};

LetterboxMeta compute_letterbox(int frame_w, int frame_h, int size);

// BGR uint8 HxWx3 → float32 CHW RGB in [0,1], size x size.
void letterbox_cpu(const uint8_t* bgr, const LetterboxMeta& m, float* chw_out);

}  // namespace nurosim
