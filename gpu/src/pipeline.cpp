#include "detect2deploy/pipeline.h"

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "detect2deploy/preprocess_gpu.h"

namespace detect2deploy {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
float event_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}
}  // namespace

Pipeline::Pipeline(const std::string& engine_path, const FrameSet& frames, const PipelineOptions& opt)
    : engine_(engine_path, opt.batch), frames_(frames), opt_(opt) {
  engine_.set_batch(opt.batch);
  const int S = engine_.input_size();
  meta_ = compute_letterbox(frames.w, frames.h, S);
  frame_bytes_ = frames.frame_bytes();
  plane3_ = static_cast<size_t>(3) * S * S;
  if (engine_.input().dtype != nvinfer1::DataType::kFLOAT)
    throw std::runtime_error("expected a float32 engine input (export keeps FP32 I/O)");

  if (engine_.has_nms_outputs()) {
    post_mode_ = "efficient_nms";
  } else {
    post_mode_ = "cpu_nms";
    for (auto& t : engine_.tensors()) {
      if (t.is_input) continue;
      num_classes_ = static_cast<int>(t.shape.d[1]) - 4;     // (B, 4 + nc, A)
      num_anchors_ = static_cast<int>(t.shape.d[2]);
    }
  }

  staging_ = std::make_unique<PinnedBuffer<uint8_t>>(frame_bytes_ * opt.batch);
  d_frames_ = std::make_unique<DeviceBuffer<uint8_t>>(frame_bytes_ * opt.batch);
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, opt.stream_priority));
  for (auto& e : ev_) CUDA_CHECK(cudaEventCreate(&e));

  // Warm up once (TensorRT lazily initialises on the first enqueue), then capture.
  enqueue_gpu(false);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (opt.cuda_graph && !try_capture_graph())
    std::fprintf(stderr, "CUDA graph capture failed for %s; running without graphs\n", engine_path.c_str());
}

Pipeline::~Pipeline() {
  if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
  for (auto& e : ev_) if (e) cudaEventDestroy(e);
  if (stream_) cudaStreamDestroy(stream_);
}

void Pipeline::enqueue_gpu(bool ev) {
  IOTensor& in = engine_.input();
  float* d_in = static_cast<float*>(in.device);
  if (ev) CUDA_CHECK(cudaEventRecord(ev_[0], stream_));
  if (opt_.gpu_pre) {
    CUDA_CHECK(cudaMemcpyAsync(d_frames_->ptr, staging_->ptr, frame_bytes_ * opt_.batch,
                               cudaMemcpyHostToDevice, stream_));
    if (ev) CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
    for (int b = 0; b < opt_.batch; ++b)
      letterbox_gpu(d_frames_->ptr + b * frame_bytes_, meta_, d_in + b * plane3_, stream_);
  } else {
    CUDA_CHECK(cudaMemcpyAsync(in.device, in.host, in.bytes(), cudaMemcpyHostToDevice, stream_));
    if (ev) CUDA_CHECK(cudaEventRecord(ev_[1], stream_));
  }
  if (ev) CUDA_CHECK(cudaEventRecord(ev_[2], stream_));
  if (!engine_.enqueue(stream_)) throw std::runtime_error("enqueueV3 failed");
  if (ev) CUDA_CHECK(cudaEventRecord(ev_[3], stream_));
  for (auto& t : engine_.tensors())
    if (!t.is_input)
      CUDA_CHECK(cudaMemcpyAsync(t.host, t.device, t.bytes(), cudaMemcpyDeviceToHost, stream_));
  if (ev) CUDA_CHECK(cudaEventRecord(ev_[4], stream_));
}

bool Pipeline::try_capture_graph() {
  // Static addresses + static shapes → the whole GPU side (copies, kernel,
  // TensorRT enqueue) replays as one graph launch: no per-kernel launch cost.
  cudaGraph_t graph = nullptr;
  if (cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
  try {
    enqueue_gpu(false);
  } catch (...) {
    cudaStreamEndCapture(stream_, &graph);
    if (graph) cudaGraphDestroy(graph);
    cudaGetLastError();
    return false;
  }
  if (cudaStreamEndCapture(stream_, &graph) != cudaSuccess || !graph) {
    cudaGetLastError();
    return false;
  }
#if CUDART_VERSION >= 12000
  const cudaError_t err = cudaGraphInstantiate(&graph_exec_, graph, 0);
#else
  const cudaError_t err = cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0);
#endif
  cudaGraphDestroy(graph);
  if (err != cudaSuccess) {
    graph_exec_ = nullptr;
    cudaGetLastError();
    return false;
  }
  return true;
}

StageTimes Pipeline::run(int first, std::vector<std::vector<Detection>>* dets) {
  StageTimes st;
  IOTensor& in = engine_.input();
  const auto t0 = Clock::now();
  {
    NvtxRange r("host_pre");
    for (int b = 0; b < opt_.batch; ++b) {
      const uint8_t* f = frames_.frame(first + b);
      if (opt_.gpu_pre) std::memcpy(staging_->ptr + b * frame_bytes_, f, frame_bytes_);
      else letterbox_cpu(f, meta_, static_cast<float*>(in.host) + b * plane3_);
    }
  }
  const auto t1 = Clock::now();
  {
    NvtxRange r("gpu");
    if (graph_exec_) {
      CUDA_CHECK(cudaEventRecord(ev_[0], stream_));
      CUDA_CHECK(cudaGraphLaunch(graph_exec_, stream_));
      CUDA_CHECK(cudaEventRecord(ev_[4], stream_));
    } else {
      enqueue_gpu(true);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
  const auto t2 = Clock::now();
  {
    NvtxRange r("post");
    std::vector<std::vector<Detection>> local;
    auto& out = dets ? *dets : local;
    out.assign(opt_.batch, {});
    for (int b = 0; b < opt_.batch; ++b) {
      if (post_mode_ == "efficient_nms") {
        IOTensor* num = engine_.tensor("num_dets");
        IOTensor* boxes = engine_.tensor("det_boxes");
        IOTensor* scores = engine_.tensor("det_scores");
        IOTensor* classes = engine_.tensor("det_classes");
        const int K = static_cast<int>(scores->shape.d[1]);
        out[b] = from_efficient_nms(static_cast<int32_t*>(num->host)[b],
                                    static_cast<float*>(boxes->host) + static_cast<size_t>(b) * K * 4,
                                    static_cast<float*>(scores->host) + static_cast<size_t>(b) * K,
                                    static_cast<int32_t*>(classes->host) + static_cast<size_t>(b) * K, meta_);
      } else {
        for (auto& t : engine_.tensors()) {
          if (t.is_input) continue;
          const float* o = static_cast<float*>(t.host) +
                           static_cast<size_t>(b) * (4 + num_classes_) * num_anchors_;
          out[b] = decode_yolov8(o, num_classes_, num_anchors_, meta_, opt_.conf, opt_.iou);
        }
      }
    }
  }
  const auto t3 = Clock::now();

  st.host_pre = ms_since(t0, t1);
  st.post = ms_since(t2, t3);
  st.e2e = ms_since(t0, t3);
  st.gpu_total = event_ms(ev_[0], ev_[4]);
  if (!graph_exec_) {
    st.h2d = event_ms(ev_[0], ev_[1]);
    st.gpu_pre = event_ms(ev_[1], ev_[2]);
    st.infer = event_ms(ev_[2], ev_[3]);
    st.d2h = event_ms(ev_[3], ev_[4]);
  }
  return st;
}

std::string device_json() {
  int dev = 0, driver = 0, runtime = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  std::ostringstream o;
  o << "{\"name\": \"" << p.name << "\", \"compute_capability\": \"" << p.major << "." << p.minor
    << "\", \"sms\": " << p.multiProcessorCount << ", \"integrated\": " << (p.integrated ? "true" : "false")
    << ", \"cuda_driver\": " << driver << ", \"cuda_runtime\": " << runtime
    << ", \"tensorrt\": \"" << trt_version() << "\"}";
  return o.str();
}

}  // namespace detect2deploy
