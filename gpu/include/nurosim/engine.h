#pragma once
// TensorRT engine build (ONNX → plan) and runtime wrapper. Targets the
// TensorRT >= 8.6 API (enqueueV3 / named I/O tensors), so it builds against
// JetPack 6 on Jetson Orin and TensorRT 10.x on x86.
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace nurosim {

class TrtLogger : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(Severity min = Severity::kWARNING) : min_(min) {}
  void log(Severity severity, const char* msg) noexcept override;
  void set_min(Severity s) { min_ = s; }

 private:
  Severity min_;
};
TrtLogger& logger();

struct BuildOptions {
  bool fp16 = false;
  bool int8 = false;            // Q/DQ (explicitly quantised) ONNX → INT8 kernels
  int min_batch = 1, opt_batch = 1, max_batch = 1;
  size_t workspace_mb = 2048;
  std::string timing_cache;     // shared across builds: rebuilds take seconds, not minutes
  int dla_core = -1;            // Jetson Orin DLA; unsupported layers fall back to the GPU
  int opt_level = 3;            // builder optimisation level (0-5)
  // Layers whose name matches are pinned to FP32 (precision constraints). Used
  // to keep the YOLOv8 head tail (box decode) out of FP16, where 0.5 px
  // coordinate steps above 512 cost small objects IoU.
  std::string fp32_layer_regex;
  // Strongly typed network: precision comes from the ONNX graph (e.g. the
  // fp16 / fp16_mixed / Q/DQ variants), not from builder flags. Optional on
  // TensorRT 10; the only mode on TensorRT >= 11, which removed kFP16 / kINT8
  // and per-layer precision APIs.
  bool strongly_typed = false;
};

bool build_engine(const std::string& onnx_path, const std::string& engine_path,
                  const BuildOptions& opt);

struct IOTensor {
  std::string name;
  bool is_input = false;
  nvinfer1::DataType dtype{};
  size_t elem_size = 0;
  size_t capacity = 0;          // bytes, sized for max batch
  void* device = nullptr;
  void* host = nullptr;         // pinned mirror
  nvinfer1::Dims shape{};       // current shape (after set_batch)
  size_t bytes() const;
};

class TrtEngine {
 public:
  explicit TrtEngine(const std::string& engine_path, int max_batch = 1);
  ~TrtEngine();
  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  void set_batch(int b);
  bool enqueue(cudaStream_t stream);
  IOTensor& input() { return io_[input_idx_]; }
  IOTensor* tensor(const std::string& name);
  std::vector<IOTensor>& tensors() { return io_; }
  bool has_nms_outputs() const;
  int batch() const { return batch_; }
  int max_batch() const { return max_batch_; }
  bool dynamic_batch() const { return dynamic_; }
  int input_size() const;       // model input H (== W)
  double load_ms() const { return load_ms_; }
  std::string describe() const;

 private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::vector<IOTensor> io_;
  int input_idx_ = 0, batch_ = 1, max_batch_ = 1;
  bool dynamic_ = false;
  nvinfer1::Dims input_max_{};
  double load_ms_ = 0;
  void refresh_shapes();
};

std::string trt_version();

}  // namespace nurosim
