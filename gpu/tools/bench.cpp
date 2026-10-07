// d2d_bench: end-to-end latency of one detector pipeline, or a detection
// dump for accuracy parity.
//
//   d2d_bench --engine m.engine --frames bench.nsfr [--pre gpu|cpu] [--cuda-graph]
//                 [--batch 1] [--warmup 50] [--iters 500] [--json out.json] [--label name]
//   d2d_bench --engine m.engine --frames eval.nsfr --dump-dets dets.json [--conf 0.01]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "detect2deploy/cli.h"
#include "detect2deploy/pipeline.h"
#include "detect2deploy/stats.h"

using namespace detect2deploy;

static int dump_detections(Pipeline& p, const FrameSet& frames, const std::string& path,
                           const std::string& engine) {
  std::ofstream f(path);
  f << "{\"engine\": \"" << engine << "\", \"post\": \"" << p.post_mode() << "\", \"frames\": [\n";
  std::vector<std::vector<Detection>> dets;
  for (int i = 0; i < frames.n; ++i) {
    p.run(i, &dets);
    f << (i ? ",\n" : "") << "[";
    for (size_t k = 0; k < dets[0].size(); ++k) {
      const auto& d = dets[0][k];
      f << (k ? ", " : "") << "[" << d.x1 << ", " << d.y1 << ", " << d.x2 << ", " << d.y2 << ", "
        << d.cls << ", " << d.score << "]";
    }
    f << "]";
  }
  f << "\n]}\n";
  std::fprintf(stderr, "wrote detections for %d frames to %s\n", frames.n, path.c_str());
  return 0;
}

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    const std::string engine = a.req("engine");
    const FrameSet frames = load_frames(a.req("frames"));
    PipelineOptions o;
    o.gpu_pre = a.str("pre", "gpu") == "gpu";
    o.cuda_graph = a.has("cuda-graph");
    o.batch = a.i("batch", 1);
    o.conf = static_cast<float>(a.f("conf", 0.25));
    o.iou = static_cast<float>(a.f("iou", 0.6));

    Pipeline p(engine, frames, o);
    std::fprintf(stderr, "%s\n%s", engine.c_str(), p.engine().describe().c_str());
    if (a.has("dump-dets")) return dump_detections(p, frames, a.str("dump-dets"), engine);

    const auto first = p.run(0);                      // first timed run after load
    const int warmup = a.i("warmup", 50), iters = a.i("iters", 500);
    for (int i = 0; i < warmup; ++i) p.run((i * o.batch) % frames.n);

    std::vector<double> e2e, host_pre, h2d, gpu_pre, infer, d2h, gpu_total, post;
    for (int i = 0; i < iters; ++i) {
      const StageTimes t = p.run((i * o.batch) % frames.n);
      e2e.push_back(t.e2e); host_pre.push_back(t.host_pre); post.push_back(t.post);
      gpu_total.push_back(t.gpu_total);
      if (!p.graph_active()) {
        h2d.push_back(t.h2d); gpu_pre.push_back(t.gpu_pre); infer.push_back(t.infer); d2h.push_back(t.d2h);
      }
    }
    const Summary s = summarize(e2e);
    std::ifstream ef(engine, std::ios::binary | std::ios::ate);

    std::ostringstream j;
    j << "{\n  \"label\": \"" << a.str("label", engine) << "\",\n  \"engine\": \"" << engine
      << "\",\n  \"engine_mb\": " << static_cast<double>(ef.tellg()) / 1e6 << ",\n  \"pre\": \"" << (o.gpu_pre ? "gpu" : "cpu")
      << "\",\n  \"post\": \"" << p.post_mode() << "\",\n  \"cuda_graph\": "
      << (p.graph_active() ? "true" : "false") << ",\n  \"batch\": " << o.batch
      << ",\n  \"warmup\": " << warmup << ",\n  \"iters\": " << iters
      << ",\n  \"device\": " << device_json()
      << ",\n  \"cold_start\": {\"engine_load_ms\": " << p.engine().load_ms()
      << ", \"first_run_ms\": " << first.e2e << "}"
      << ",\n  \"throughput_fps\": " << o.batch * 1000.0 / s.p50
      << ",\n  \"e2e\": " << to_json(s) << ",\n  \"stages\": {"
      << "\"host_pre\": " << to_json(summarize(host_pre)) << ", \"post\": " << to_json(summarize(post))
      << ", \"gpu_total\": " << to_json(summarize(gpu_total));
    if (!p.graph_active())
      j << ", \"h2d\": " << to_json(summarize(h2d)) << ", \"gpu_pre\": " << to_json(summarize(gpu_pre))
        << ", \"infer\": " << to_json(summarize(infer)) << ", \"d2h\": " << to_json(summarize(d2h));
    j << "}\n}\n";

    std::printf("%-40s pre=%s post=%-13s graph=%d bs=%d  e2e p50 %.3f ms  p99 %.3f ms  max %.3f ms  %.0f fps\n",
                a.str("label", engine).c_str(), o.gpu_pre ? "gpu" : "cpu", p.post_mode().c_str(),
                p.graph_active(), o.batch, s.p50, s.p99, s.max, o.batch * 1000.0 / s.p50);
    if (a.has("json")) std::ofstream(a.str("json")) << j.str();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
