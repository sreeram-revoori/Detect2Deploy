#pragma once
// One detector pipeline on one CUDA stream:
//   host frame → [pinned staging] → H2D → preprocess → TensorRT → D2H → post
// Preprocessing runs on the GPU (fused letterbox kernel) or on the host (the
// CPU reference, i.e. what the Python path does). Postprocessing is either the
// in-engine EfficientNMS plugin or CPU decode + NMS. The GPU part can be
// captured once into a CUDA graph and replayed.
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "nurosim/cuda_utils.h"
#include "nurosim/engine.h"
#include "nurosim/frames.h"
#include "nurosim/letterbox.h"
#include "nurosim/postprocess.h"

namespace nurosim {

struct StageTimes {
  double host_pre = 0, h2d = 0, gpu_pre = 0, infer = 0, d2h = 0, gpu_total = 0, post = 0, e2e = 0;
};

struct PipelineOptions {
  bool gpu_pre = true;
  bool cuda_graph = false;
  int batch = 1;
  float conf = 0.25f;           // CPU-decode path only (EfficientNMS has its thresholds baked in)
  float iou = 0.6f;
  int stream_priority = 0;      // lower number = higher priority (cudaDeviceGetStreamPriorityRange)
};

class Pipeline {
 public:
  Pipeline(const std::string& engine_path, const FrameSet& frames, const PipelineOptions& opt);
  ~Pipeline();

  // Runs frames [first, first + batch). Detections per frame if dets != nullptr.
  StageTimes run(int first_frame, std::vector<std::vector<Detection>>* dets = nullptr);

  bool graph_active() const { return graph_exec_ != nullptr; }
  const std::string& post_mode() const { return post_mode_; }
  TrtEngine& engine() { return engine_; }
  cudaStream_t stream() const { return stream_; }

 private:
  void enqueue_gpu(bool with_events);
  bool try_capture_graph();

  TrtEngine engine_;
  const FrameSet& frames_;
  PipelineOptions opt_;
  LetterboxMeta meta_{};
  std::string post_mode_;
  size_t frame_bytes_ = 0, plane3_ = 0;
  std::unique_ptr<PinnedBuffer<uint8_t>> staging_;
  std::unique_ptr<DeviceBuffer<uint8_t>> d_frames_;
  cudaStream_t stream_ = nullptr;
  cudaEvent_t ev_[5] = {};
  cudaGraphExec_t graph_exec_ = nullptr;
  int num_classes_ = 0, num_anchors_ = 0;
};

std::string device_json();   // GPU name, SMs, driver/runtime/TensorRT versions

}  // namespace nurosim
