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
#include <functional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/time/clock.h"
#include "ray/common/id.h"
#include "ray/common/ray_object.h"
#include "ray/raylet/aom_types.h"

namespace ray {
namespace raylet {

    class AOMPolicy {
        public:
        virtual ~AOMPolicy() = default;

        virtual std::vector<ObjectID> SelectEvictionCandidates(
            const absl::flat_hash_map<ObjectID, ObjectAccessStats> &access_stats,
            const absl::flat_hash_map<ObjectID, std::unique_ptr<RayObject>> &pinned_objects,
            std::function<bool(const ObjectID &)> is_spillable,
            int64_t bytes_to_free) = 0;
        
        virtual std::string Name() const = 0;
    };

    class FrequencyWeightedPolicy : public AOMPolicy {
        public:
        explicit FrequencyWeightedPolicy(double decay_rate) : decay_rate_(decay_rate) {}

        std::vector<ObjectID> SelectEvictionCandidates(
            const absl::flat_hash_map<ObjectID, ObjectAccessStats> &access_stats,
            const absl::flat_hash_map<ObjectID, std::unique_ptr<RayObject>> &pinned_objects,
            std::function<bool(const ObjectID &)> is_spillable,
            int64_t bytes_to_free) override;
        
        std::string Name() const override { return "FrequencyWeighted"; }
        
        private:
        double decay_rate_;
    };
}  // namespace raylet
}  // namespace ray