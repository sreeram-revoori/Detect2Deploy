#include "detect2deploy/stats.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace detect2deploy {

double percentile(std::vector<double> v, double q) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = q / 100.0 * (v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx)), hi = static_cast<size_t>(std::ceil(idx));
  return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

Summary summarize(const std::vector<double>& a) {
  Summary s;
  s.n = static_cast<int>(a.size());
  if (a.empty()) return s;
  s.mean = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
  double var = 0;
  for (double x : a) var += (x - s.mean) * (x - s.mean);
  s.std = std::sqrt(var / a.size());
  s.p50 = percentile(a, 50);
  s.p90 = percentile(a, 90);
  s.p99 = percentile(a, 99);
  s.max = *std::max_element(a.begin(), a.end());
  s.jitter_p99_p50 = s.p99 - s.p50;
  return s;
}

std::string to_json(const Summary& s) {
  std::ostringstream o;
  o.precision(5);
  o << "{\"n\": " << s.n << ", \"mean\": " << s.mean << ", \"std\": " << s.std
    << ", \"p50\": " << s.p50 << ", \"p90\": " << s.p90 << ", \"p99\": " << s.p99
    << ", \"max\": " << s.max << ", \"jitter_p99_p50\": " << s.jitter_p99_p50 << "}";
  return o.str();
}

}  // namespace detect2deploy
