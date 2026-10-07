#pragma once
// Minimal --key value / --flag argument parsing for the tools.
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

namespace nurosim {

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k.rfind("--", 0) != 0) throw std::runtime_error("unexpected argument: " + k);
      k = k.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) kv_[k] = argv[++i];
      else kv_[k] = "1";
    }
  }
  bool has(const std::string& k) const { return kv_.count(k) > 0; }
  std::string str(const std::string& k, const std::string& def = "") const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  std::string req(const std::string& k) const {
    if (!has(k)) throw std::runtime_error("missing required --" + k);
    return kv_.at(k);
  }
  int i(const std::string& k, int def) const { return has(k) ? std::atoi(kv_.at(k).c_str()) : def; }
  double f(const std::string& k, double def) const { return has(k) ? std::atof(kv_.at(k).c_str()) : def; }

 private:
  std::map<std::string, std::string> kv_;
};

}  // namespace nurosim
