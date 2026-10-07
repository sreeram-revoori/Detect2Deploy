// d2d_build_engine: ONNX → TensorRT engine.
//   d2d_build_engine --onnx m.onnx --out m.engine [--fp16] [--int8]
//       [--max-batch 8] [--opt-batch 1] [--workspace-mb 2048]
//       [--timing-cache trt.cache] [--dla 0] [--opt-level 3]
//       [--fp32-layers '^/model\.22/(?!cv2|cv3)']   # keep the head tail in FP32
//       [--strongly-typed]     # precision from the ONNX graph (always on with TensorRT >= 11)
//   d2d_build_engine --version
#include <cstdio>
#include <exception>

#include "detect2deploy/cli.h"
#include "detect2deploy/engine.h"

int main(int argc, char** argv) {
  try {
    detect2deploy::Args a(argc, argv);
    if (a.has("version")) {
      std::printf("%s\n", detect2deploy::trt_version().c_str());
      return 0;
    }
    detect2deploy::BuildOptions o;
    o.fp16 = a.has("fp16");
    o.int8 = a.has("int8");
    o.max_batch = a.i("max-batch", 1);
    o.opt_batch = a.i("opt-batch", 1);
    o.min_batch = a.i("min-batch", 1);
    o.workspace_mb = static_cast<size_t>(a.i("workspace-mb", 2048));
    o.timing_cache = a.str("timing-cache");
    o.dla_core = a.i("dla", -1);
    o.opt_level = a.i("opt-level", 3);
    o.fp32_layer_regex = a.str("fp32-layers");
    o.strongly_typed = a.has("strongly-typed");
    if (a.has("verbose")) detect2deploy::logger().set_min(nvinfer1::ILogger::Severity::kINFO);
    std::fprintf(stderr, "TensorRT %s\n", detect2deploy::trt_version().c_str());
    return detect2deploy::build_engine(a.req("onnx"), a.req("out"), o) ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
