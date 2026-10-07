#pragma once
// Tiny dependency-free test harness.
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace check {
inline int& failures() { static int f = 0; return f; }
inline std::vector<std::pair<std::string, std::function<void()>>>& registry() {
  static std::vector<std::pair<std::string, std::function<void()>>> r;
  return r;
}
struct Reg { Reg(const char* n, std::function<void()> f) { registry().emplace_back(n, std::move(f)); } };
inline int run_all() {
  for (auto& [name, fn] : registry()) {
    const int before = failures();
    fn();
    std::printf("%s %s\n", failures() == before ? "PASS" : "FAIL", name.c_str());
  }
  std::printf("%zu tests, %d failed checks\n", registry().size(), failures());
  return failures() ? 1 : 0;
}
}  // namespace check

#define TEST(name) static void name(); static check::Reg reg_##name(#name, name); static void name()
#define CHECK(c) do { if (!(c)) { std::printf("  %s:%d CHECK(%s)\n", __FILE__, __LINE__, #c); ++check::failures(); } } while (0)
#define CHECK_NEAR(a, b, tol) do { double a_ = (a), b_ = (b); if (std::fabs(a_ - b_) > (tol)) { \
  std::printf("  %s:%d %s=%g vs %s=%g\n", __FILE__, __LINE__, #a, a_, #b, b_); ++check::failures(); } } while (0)
