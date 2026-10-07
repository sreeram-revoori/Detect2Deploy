#include "detect2deploy/engine.h"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "detect2deploy/cuda_utils.h"

namespace detect2deploy {

using namespace nvinfer1;

void TrtLogger::log(Severity severity, const char* msg) noexcept {
  if (severity > min_) return;
  static const char* tags[] = {"INTERNAL_ERROR", "ERROR", "WARNING", "INFO", "VERBOSE"};
  std::fprintf(stderr, "[TRT %s] %s\n", tags[static_cast<int>(severity)], msg);
}

TrtLogger& logger() {
  static TrtLogger l;
  return l;
}

std::string trt_version() {
  const int v = getInferLibVersion();
  // TensorRT >= 10 encodes major*10000 + minor*100 + patch; 8.x used major*1000 + minor*100 + patch
  const int major = v >= 10000 ? v / 10000 : v / 1000;
  const int minor = v >= 10000 ? (v / 100) % 100 : (v / 100) % 10;
  return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(v % 100);
}

namespace {

size_t dtype_size(DataType t) {
  switch (t) {
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT32: return 4;
    case DataType::kINT8: return 1;
    case DataType::kBOOL: return 1;
    case DataType::kUINT8: return 1;
#if NV_TENSORRT_MAJOR >= 10
    case DataType::kBF16: return 2;
    case DataType::kINT64: return 8;
#endif
    default: throw std::runtime_error("unsupported TensorRT tensor dtype");
  }
}

size_t volume(const Dims& d) {
  size_t v = 1;
  for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i] < 0 ? 0 : d.d[i]);
  return v;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return {};
  std::vector<char> buf(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
  return buf;
}

void write_file(const std::string& path, const void* data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

}  // namespace

size_t IOTensor::bytes() const { return volume(shape) * elem_size; }

bool build_engine(const std::string& onnx_path, const std::string& engine_path,
                  const BuildOptions& opt) {
  auto& log = logger();
  initLibNvInferPlugins(&log, "");    // EfficientNMS_TRT et al.

#if NV_TENSORRT_MAJOR >= 11
  const bool strongly_typed = true;
#else
  const bool strongly_typed = opt.strongly_typed;
#endif
  if (strongly_typed && (opt.fp16 || opt.int8 || !opt.fp32_layer_regex.empty())) {
    std::fprintf(stderr,
                 "strongly typed build (TensorRT %s): precision comes from the ONNX graph. Build FP16 "
                 "engines from the FP16 ONNX variants and INT8 from the Q/DQ model, without "
                 "--fp16 / --int8 / --fp32-layers.\n", trt_version().c_str());
    return false;
  }

  std::unique_ptr<IBuilder> builder{createInferBuilder(log)};
  uint32_t flags = 0;
#if NV_TENSORRT_MAJOR < 10
  flags |= 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#elif NV_TENSORRT_MAJOR == 10
  if (strongly_typed) flags |= 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#endif
  std::unique_ptr<INetworkDefinition> network{builder->createNetworkV2(flags)};
  std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, log)};
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
    for (int i = 0; i < parser->getNbErrors(); ++i)
      std::fprintf(stderr, "ONNX parse error: %s\n", parser->getError(i)->desc());
    return false;
  }

  std::unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, opt.workspace_mb << 20);
  config->setBuilderOptimizationLevel(opt.opt_level);
#if NV_TENSORRT_MAJOR < 11
  if (opt.fp16) config->setFlag(BuilderFlag::kFP16);
  if (opt.int8) {
    // Q/DQ model: scales come from the graph (explicit quantisation); FP16
    // is allowed for the layers left unquantised (the FP32 head tail).
    config->setFlag(BuilderFlag::kINT8);
    config->setFlag(BuilderFlag::kFP16);
  }
#endif
  if (opt.dla_core >= 0) {
    config->setDefaultDeviceType(DeviceType::kDLA);
    config->setDLACore(opt.dla_core);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
#if NV_TENSORRT_MAJOR < 11
    if (!opt.int8 && !strongly_typed) config->setFlag(BuilderFlag::kFP16);   // DLA has no FP32
#endif
  }

#if NV_TENSORRT_MAJOR < 11
  if (!opt.fp32_layer_regex.empty()) {
    const std::regex re(opt.fp32_layer_regex);
    int pinned = 0;
    for (int i = 0; i < network->getNbLayers(); ++i) {
      ILayer* layer = network->getLayer(i);
      if (!std::regex_search(layer->getName(), re)) continue;
      bool all_float = true;
      for (int o = 0; o < layer->getNbOutputs(); ++o)
        all_float &= layer->getOutput(o)->getType() == DataType::kFLOAT;
      if (!all_float) continue;              // shape / index tensors keep their type
      layer->setPrecision(DataType::kFLOAT);
      for (int o = 0; o < layer->getNbOutputs(); ++o) layer->setOutputType(o, DataType::kFLOAT);
      ++pinned;
    }
    config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
    std::fprintf(stderr, "pinned %d layers to FP32 (regex %s)\n", pinned, opt.fp32_layer_regex.c_str());
  }
#endif

  ITensor* in = network->getInput(0);
  Dims d = in->getDimensions();
  if (d.d[0] == -1) {
    IOptimizationProfile* prof = builder->createOptimizationProfile();
    Dims mn = d, op = d, mx = d;
    mn.d[0] = opt.min_batch;
    op.d[0] = opt.opt_batch;
    mx.d[0] = opt.max_batch;
    prof->setDimensions(in->getName(), OptProfileSelector::kMIN, mn);
    prof->setDimensions(in->getName(), OptProfileSelector::kOPT, op);
    prof->setDimensions(in->getName(), OptProfileSelector::kMAX, mx);
    config->addOptimizationProfile(prof);
  }

  std::vector<char> cache_blob;
  if (!opt.timing_cache.empty()) cache_blob = read_file(opt.timing_cache);
  std::unique_ptr<ITimingCache> cache{config->createTimingCache(cache_blob.data(), cache_blob.size())};
  config->setTimingCache(*cache, false);

  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
  if (!plan) return false;
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  write_file(engine_path, plan->data(), plan->size());
  std::fprintf(stderr, "built %s in %.1fs (%.1f MB)\n", engine_path.c_str(), secs, plan->size() / 1e6);

  if (!opt.timing_cache.empty()) {
    std::unique_ptr<IHostMemory> blob{config->getTimingCache()->serialize()};
    write_file(opt.timing_cache, blob->data(), blob->size());
  }
  return true;
}

TrtEngine::TrtEngine(const std::string& engine_path, int max_batch) {
  const auto t0 = std::chrono::steady_clock::now();
  auto& log = logger();
  initLibNvInferPlugins(&log, "");
  const std::vector<char> plan = read_file(engine_path);
  if (plan.empty()) throw std::runtime_error("cannot read engine " + engine_path);
  runtime_.reset(createInferRuntime(log));
  engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine_) throw std::runtime_error("cannot deserialize " + engine_path);
  context_.reset(engine_->createExecutionContext());

  const int n = engine_->getNbIOTensors();
  for (int i = 0; i < n; ++i) {
    IOTensor t;
    t.name = engine_->getIOTensorName(i);
    t.is_input = engine_->getTensorIOMode(t.name.c_str()) == TensorIOMode::kINPUT;
    t.dtype = engine_->getTensorDataType(t.name.c_str());
    t.elem_size = dtype_size(t.dtype);
    if (t.is_input) input_idx_ = static_cast<int>(io_.size());
    io_.push_back(t);
  }

  const char* in_name = io_[input_idx_].name.c_str();
  Dims shape = engine_->getTensorShape(in_name);
  dynamic_ = shape.d[0] == -1;
  if (dynamic_) {
    input_max_ = engine_->getProfileShape(in_name, 0, OptProfileSelector::kMAX);
    max_batch_ = std::min(max_batch, static_cast<int>(input_max_.d[0]));
    input_max_.d[0] = max_batch_;
    context_->setInputShape(in_name, input_max_);
  } else {
    input_max_ = shape;
    max_batch_ = static_cast<int>(shape.d[0]);
  }

  // Allocate every tensor once at max batch, bind addresses once.
  for (auto& t : io_) {
    t.shape = context_->getTensorShape(t.name.c_str());
    t.capacity = std::max<size_t>(t.bytes(), 1);
    CUDA_CHECK(cudaMalloc(&t.device, t.capacity));
    CUDA_CHECK(cudaMallocHost(&t.host, t.capacity));
    context_->setTensorAddress(t.name.c_str(), t.device);
  }
  set_batch(1);
  load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

TrtEngine::~TrtEngine() {
  for (auto& t : io_) {
    if (t.device) cudaFree(t.device);
    if (t.host) cudaFreeHost(t.host);
  }
}

void TrtEngine::refresh_shapes() {
  for (auto& t : io_) t.shape = context_->getTensorShape(t.name.c_str());
}

void TrtEngine::set_batch(int b) {
  if (b > max_batch_) throw std::runtime_error("batch exceeds engine max batch");
  if (dynamic_) {
    Dims d = input_max_;
    d.d[0] = b;
    context_->setInputShape(io_[input_idx_].name.c_str(), d);
  } else if (b != max_batch_) {
    throw std::runtime_error("static-batch engine: batch must equal " + std::to_string(max_batch_));
  }
  batch_ = b;
  refresh_shapes();
}

bool TrtEngine::enqueue(cudaStream_t stream) { return context_->enqueueV3(stream); }

IOTensor* TrtEngine::tensor(const std::string& name) {
  for (auto& t : io_)
    if (t.name == name) return &t;
  return nullptr;
}

bool TrtEngine::has_nms_outputs() const {
  for (const auto& t : io_)
    if (t.name == "num_dets") return true;
  return false;
}

int TrtEngine::input_size() const { return static_cast<int>(input_max_.d[2]); }

std::string TrtEngine::describe() const {
  std::ostringstream o;
  for (const auto& t : io_) {
    o << (t.is_input ? "  in  " : "  out ") << t.name << " [";
    for (int i = 0; i < t.shape.nbDims; ++i) o << (i ? "," : "") << t.shape.d[i];
    o << "] dtype=" << static_cast<int>(t.dtype) << "\n";
  }
  return o.str();
}

}  // namespace detect2deploy
