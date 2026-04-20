// Copyright 2026 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>

namespace ray {
namespace raylet {

/// Per-object access statistics tracked by the AOM Observer.
struct ObjectAccessStats {
  int64_t created_at_ns = 0;
  int64_t last_access_ns = 0;
  uint64_t access_count = 0;
  size_t object_size = 0;
  bool was_restored = false;
  
  std::deque<int64_t> recent_access_times_ns;

  double ComputeTemperature(int64_t now_ns, double decay_rate) const {
    double recency = std::exp(-decay_rate * static_cast<double>(now_ns - last_access_ns));
    double frequency = static_cast<double>(access_count);
    double size_cost = std::log1p(static_cast<double>(object_size));
    double restore_bonus = was_restored ? 1.5 : 1.0;
    return (frequency * recency * size_cost) * restore_bonus;
  }

  void RecordAccess(int64_t now_ns, int max_k = 2) {
    recent_access_times_ns.push_front(now_ns);
    if (static_cast<int>(recent_access_times_ns.size()) > max_k) {
      recent_access_times_ns.pop_back();
    }
  }

  int64_t GetKthAccessTime(int k) const {
    if (k < 0 || k >= static_cast<int>(recent_access_times_ns.size())) {
      return created_at_ns;
    }
    return recent_access_times_ns[k];
  }
};

}  // namespace raylet
}  // namespace ray
