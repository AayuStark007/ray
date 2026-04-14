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

#include "ray/raylet/aom_policy.h"

#include "ray/util/logging.h"

namespace ray {
namespace raylet {
    std::vector<ObjectID> FrequencyWeightedPolicy::SelectEvictionCandidates(
        const absl::flat_hash_map<ObjectID, ObjectAccessStats> &access_stats,
        const absl::flat_hash_map<ObjectID, std::unique_ptr<RayObject>> &pinned_objects,
        std::function<bool(const ObjectID &)> is_spillable,
        int64_t bytes_to_free) {


            if (bytes_to_free <= 0 || pinned_objects.empty()) {
                return {};
            }

            auto now_ns = absl::GetCurrentTimeNanos();

            // Build a vector of (temperature, object_id) for all spillable pinned objects.
            std::vector<std::pair<double, ObjectID>> scored_objects;
            scored_objects.reserve(pinned_objects.size());
            for (const auto &[object_id, ray_object] : pinned_objects) {
                if (!is_spillable(object_id)) {
                    continue;
                }
                auto stats_it = access_stats.find(object_id);
                double temperature = 0.0;
                if (stats_it != access_stats.end()) {
                    temperature = stats_it->second.ComputeTemperature(now_ns, decay_rate_);
                }
                scored_objects.emplace_back(temperature, object_id);
            }

            // Sort by temperature ascending (coldest first).
            std::sort(scored_objects.begin(), scored_objects.end(), 
                        [](const auto &a, const auto &b) { return a.first < b.first; });
            
            // Collect the coldest objects until we have enough bytes.
            std::vector<ObjectID> candidates;
            int64_t accumulated = 0;
            for (const auto &[temp, object_id] : scored_objects) {
                if (accumulated >= bytes_to_free) {
                    break;
                }

                auto it = pinned_objects.find(object_id);
                if (it != pinned_objects.end()) {
                    accumulated += it->second->GetSize();
                    candidates.push_back(object_id);
                }
            }

            RAY_LOG(DEBUG) << "AOM FrequencyWeightedPolicy: selected " << candidates.size()
                           << " objects (" << accumulated << " bytes) to spill"
                           << " (target: " << bytes_to_free << " bytes)";
            
            return candidates;
    }
}  // namespace raylet
}  // namespace ray