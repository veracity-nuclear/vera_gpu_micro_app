// Minimal Kokkos Tools profiler: accumulates wall time per kernel name, per region, and deep copies.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
using clk = std::chrono::steady_clock;
struct Acc { double t = 0; uint64_t n = 0; };
static std::map<std::string, Acc> kernels, regions;
static std::map<uint64_t, std::pair<std::string, clk::time_point>> live;
static std::vector<std::pair<std::string, clk::time_point>> regstack;
static uint64_t next_id = 1;
static clk::time_point t_init;
static Acc deepcopy; static clk::time_point dc_start;
extern "C" {
void kokkosp_init_library(int, uint64_t, uint32_t, void*) { t_init = clk::now(); }
static void begin(const char* name, uint64_t* kID) { *kID = next_id++; live[*kID] = {name, clk::now()}; }
static void end(uint64_t kID) { auto it = live.find(kID); if (it == live.end()) return;
  auto& a = kernels[it->second.first]; a.t += std::chrono::duration<double>(clk::now() - it->second.second).count(); a.n++; live.erase(it); }
void kokkosp_begin_parallel_for(const char* n, uint32_t, uint64_t* k) { begin(n, k); }
void kokkosp_end_parallel_for(uint64_t k) { end(k); }
void kokkosp_begin_parallel_scan(const char* n, uint32_t, uint64_t* k) { begin(n, k); }
void kokkosp_end_parallel_scan(uint64_t k) { end(k); }
void kokkosp_begin_parallel_reduce(const char* n, uint32_t, uint64_t* k) { begin(n, k); }
void kokkosp_end_parallel_reduce(uint64_t k) { end(k); }
void kokkosp_push_profile_region(const char* n) { regstack.push_back({n, clk::now()}); }
void kokkosp_pop_profile_region() { if (regstack.empty()) return; auto& r = regstack.back();
  auto& a = regions[r.first]; a.t += std::chrono::duration<double>(clk::now() - r.second).count(); a.n++; regstack.pop_back(); }
void kokkosp_begin_deep_copy(void*, const char*, const void*, void*, const char*, const void*, uint64_t) { dc_start = clk::now(); }
void kokkosp_end_deep_copy() { deepcopy.t += std::chrono::duration<double>(clk::now() - dc_start).count(); deepcopy.n++; }
void kokkosp_finalize_library() {
  double total = std::chrono::duration<double>(clk::now() - t_init).count();
  fprintf(stderr, "\n==== kptimer: total wall since Kokkos init = %.3f s ====\n", total);
  auto dump = [](const char* title, std::map<std::string, Acc>& m) {
    std::vector<std::pair<std::string, Acc>> v(m.begin(), m.end());
    std::sort(v.begin(), v.end(), [](auto& a, auto& b){ return a.second.t > b.second.t; });
    fprintf(stderr, "-- %s --\n%12s %10s %12s  name\n", title, "total(s)", "calls", "avg(us)");
    for (auto& p : v) fprintf(stderr, "%12.4f %10lu %12.1f  %s\n", p.second.t, (unsigned long)p.second.n, 1e6*p.second.t/std::max<uint64_t>(1,p.second.n), p.first.c_str());
  };
  dump("kernels", kernels); dump("regions", regions);
  fprintf(stderr, "-- deep_copy: %.4f s over %lu calls --\n", deepcopy.t, (unsigned long)deepcopy.n);
}
}
