#pragma once
#include <string>
#include <vector>

namespace nurosim {

struct Summary {
  int n = 0;
  double mean = 0, std = 0, p50 = 0, p90 = 0, p99 = 0, max = 0, jitter_p99_p50 = 0;
};

// Linear-interpolated percentile, identical to numpy.percentile's default.
double percentile(std::vector<double> v, double q);
Summary summarize(const std::vector<double>& samples_ms);
std::string to_json(const Summary& s);

}  // namespace nurosim
