// nurosim_multistream: two models sharing one GPU.
//
// Model A is the latency-critical detector, fed periodically like a camera
// (default 30 Hz, batch 1). Model B is a heavy background workload run
// back-to-back (default: the same network at batch 8). Phases:
//   a_solo, b_solo, concurrent (equal stream priority), concurrent (A high priority)
// and for each: A's latency distribution, A's deadline misses (latency > frame
// period) and B's throughput. Shows how much a co-located model inflates the
// critical model's tail, and how much stream priority buys back.
//
//   nurosim_multistream --engine-a a.engine --engine-b b.engine --frames bench.nsfr
//                       [--rate-a-hz 30] [--batch-b 8] [--duration-s 8] [--json out.json]
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "nurosim/cli.h"
#include "nurosim/pipeline.h"
#include "nurosim/stats.h"

using namespace nurosim;
using Clock = std::chrono::steady_clock;

struct PhaseResult {
  std::string name;
  Summary a;
  int a_frames = 0, a_misses = 0;
  double b_fps = 0;
};

static std::vector<double> run_periodic(Pipeline& p, int frames_n, double hz, double secs, int& misses) {
  std::vector<double> lat;
  const auto period = std::chrono::duration<double>(1.0 / hz);
  const auto end = Clock::now() + std::chrono::duration<double>(secs);
  auto next = Clock::now();
  int i = 0;
  while (Clock::now() < end) {
    const StageTimes t = p.run(i++ % frames_n);
    lat.push_back(t.e2e);
    if (t.e2e > 1000.0 / hz) ++misses;
    next += std::chrono::duration_cast<Clock::duration>(period);
    if (next > Clock::now()) std::this_thread::sleep_until(next);
    else next = Clock::now();               // overran: don't try to catch up
  }
  return lat;
}

static double run_saturating(Pipeline& p, int frames_n, int batch, std::atomic<bool>& stop) {
  int iters = 0;
  const auto t0 = Clock::now();
  while (!stop.load()) p.run((iters++ * batch) % frames_n);
  const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
  return iters * batch / secs;
}

int main(int argc, char** argv) {
  try {
    Args a(argc, argv);
    const FrameSet frames = load_frames(a.req("frames"));
    const double hz = a.f("rate-a-hz", 30), secs = a.f("duration-s", 8);
    const int batch_b = a.i("batch-b", 8);
    int prio_low = 0, prio_high = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high));

    auto make = [&](const std::string& key, int batch, int prio) {
      PipelineOptions o;
      o.gpu_pre = true;
      o.batch = batch;
      o.stream_priority = prio;
      return std::make_unique<Pipeline>(a.req(key), frames, o);
    };

    std::vector<PhaseResult> results;
    {   // A alone
      auto pa = make("engine-a", 1, prio_low);
      for (int i = 0; i < 50; ++i) pa->run(i);
      PhaseResult r{"a_solo"};
      auto lat = run_periodic(*pa, frames.n, hz, secs, r.a_misses);
      r.a = summarize(lat);
      r.a_frames = static_cast<int>(lat.size());
      results.push_back(r);
    }
    {   // B alone
      auto pb = make("engine-b", batch_b, prio_low);
      for (int i = 0; i < 10; ++i) pb->run(0);
      std::atomic<bool> stop{false};
      std::thread timer([&] { std::this_thread::sleep_for(std::chrono::duration<double>(secs)); stop = true; });
      PhaseResult r{"b_solo"};
      r.b_fps = run_saturating(*pb, frames.n, batch_b, stop);
      timer.join();
      results.push_back(r);
    }
    for (bool prio : {false, true}) {   // together
      auto pa = make("engine-a", 1, prio ? prio_high : prio_low);
      auto pb = make("engine-b", batch_b, prio_low);
      for (int i = 0; i < 20; ++i) { pa->run(i); pb->run(0); }
      std::atomic<bool> stop{false};
      double b_fps = 0;
      std::thread tb([&] { b_fps = run_saturating(*pb, frames.n, batch_b, stop); });
      PhaseResult r{prio ? "concurrent_a_high_priority" : "concurrent_equal_priority"};
      auto lat = run_periodic(*pa, frames.n, hz, secs, r.a_misses);
      stop = true;
      tb.join();
      r.a = summarize(lat);
      r.a_frames = static_cast<int>(lat.size());
      r.b_fps = b_fps;
      results.push_back(r);
    }

    std::ostringstream j;
    j << "{\n  \"engine_a\": \"" << a.req("engine-a") << "\", \"engine_b\": \"" << a.req("engine-b")
      << "\", \"rate_a_hz\": " << hz << ", \"batch_b\": " << batch_b << ", \"duration_s\": " << secs
      << ",\n  \"stream_priority_range\": [" << prio_low << ", " << prio_high << "]"
      << ",\n  \"device\": " << device_json() << ",\n  \"phases\": [\n";
    std::printf("%-28s %9s %9s %9s %8s %9s\n", "phase", "A p50", "A p99", "A max", "A miss", "B fps");
    for (size_t i = 0; i < results.size(); ++i) {
      const auto& r = results[i];
      j << "    {\"name\": \"" << r.name << "\", \"a\": " << to_json(r.a) << ", \"a_frames\": " << r.a_frames
        << ", \"a_deadline_misses\": " << r.a_misses << ", \"b_fps\": " << r.b_fps << "}"
        << (i + 1 < results.size() ? ",\n" : "\n");
      std::printf("%-28s %9.3f %9.3f %9.3f %8d %9.1f\n", r.name.c_str(), r.a.p50, r.a.p99, r.a.max,
                  r.a_misses, r.b_fps);
    }
    j << "  ]\n}\n";
    if (a.has("json")) std::ofstream(a.str("json")) << j.str();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
