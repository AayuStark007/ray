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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "ray/common/buffer.h"
#include "ray/common/id.h"
#include "ray/common/ray_object.h"
#include "ray/raylet/aom_types.h"

namespace ray {
namespace raylet {

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// A minimal Buffer implementation for use in tests. Owns a vector of bytes
/// so that RayObject::GetSize() returns the desired value.
class FakeBuffer : public Buffer {
 public:
  explicit FakeBuffer(size_t size) : data_(size, 0) {}
  uint8_t *Data() const override { return const_cast<uint8_t *>(data_.data()); }
  size_t Size() const override { return data_.size(); }
  bool OwnsData() const override { return true; }
  bool IsPlasmaBuffer() const override { return false; }

 private:
  std::vector<uint8_t> data_;
};

/// Helper: create a pinned_objects map with N objects of the given size.
/// Returns the ordered list of ObjectIDs (in creation order).
static std::pair<std::vector<ObjectID>,
                 absl::flat_hash_map<ObjectID, std::unique_ptr<RayObject>>>
MakePinnedObjects(int count, size_t object_size) {
  std::vector<ObjectID> ids;
  absl::flat_hash_map<ObjectID, std::unique_ptr<RayObject>> pinned;
  for (int i = 0; i < count; ++i) {
    ObjectID id = ObjectID::FromRandom();
    ids.push_back(id);
    auto buf = std::make_shared<FakeBuffer>(object_size);
    auto obj =
        std::make_unique<RayObject>(buf, nullptr, std::vector<rpc::ObjectReference>());
    pinned.emplace(id, std::move(obj));
  }
  return {ids, std::move(pinned)};
}

/// An is_spillable that always returns true.
static bool AllSpillable(const ObjectID &) { return true; }

// =====================================================================
// ObjectAccessStats unit tests
// =====================================================================

class ObjectAccessStatsTest : public ::testing::Test {};

TEST_F(ObjectAccessStatsTest, DefaultValuesAreZero) {
  ObjectAccessStats stats;
  EXPECT_EQ(stats.created_at_ns, 0);
  EXPECT_EQ(stats.last_access_ns, 0);
  EXPECT_EQ(stats.access_count, 0u);
  EXPECT_EQ(stats.object_size, 0u);
  EXPECT_FALSE(stats.was_restored);
  EXPECT_TRUE(stats.recent_access_times_ns.empty());
}

TEST_F(ObjectAccessStatsTest, ComputeTemperatureIncreasesWithAccessCount) {
  ObjectAccessStats cold;
  cold.created_at_ns = 1000;
  cold.last_access_ns = 1000;
  cold.access_count = 1;
  cold.object_size = 100;

  ObjectAccessStats hot = cold;
  hot.access_count = 10;

  int64_t now_ns = 1000;  // Same as last_access_ns → recency ~1.0
  double decay = 1e-9;
  EXPECT_GT(hot.ComputeTemperature(now_ns, decay),
            cold.ComputeTemperature(now_ns, decay));
}

TEST_F(ObjectAccessStatsTest, ComputeTemperatureDecreasesWithStaleness) {
  ObjectAccessStats stats;
  stats.created_at_ns = 0;
  stats.last_access_ns = 0;
  stats.access_count = 5;
  stats.object_size = 1024;

  double decay = 1e-9;
  double recent = stats.ComputeTemperature(/*now_ns=*/0, decay);  // just accessed
  double stale = stats.ComputeTemperature(/*now_ns=*/2000000000LL, decay);  // 2s later
  EXPECT_GT(recent, stale);
}

TEST_F(ObjectAccessStatsTest, RestoredObjectGetsTemperatureBonus) {
  ObjectAccessStats normal;
  normal.created_at_ns = 0;
  normal.last_access_ns = 0;
  normal.access_count = 3;
  normal.object_size = 512;
  normal.was_restored = false;

  ObjectAccessStats restored = normal;
  restored.was_restored = true;

  double decay = 1e-9;
  int64_t now_ns = 0;
  EXPECT_GT(restored.ComputeTemperature(now_ns, decay),
            normal.ComputeTemperature(now_ns, decay));
  // The bonus is exactly 1.5x
  EXPECT_DOUBLE_EQ(restored.ComputeTemperature(now_ns, decay),
                   normal.ComputeTemperature(now_ns, decay) * 1.5);
}

TEST_F(ObjectAccessStatsTest, RecordAccessMaintainsBoundedDeque) {
  ObjectAccessStats stats;
  stats.RecordAccess(100, /*max_k=*/3);
  stats.RecordAccess(200, /*max_k=*/3);
  stats.RecordAccess(300, /*max_k=*/3);
  EXPECT_EQ(stats.recent_access_times_ns.size(), 3u);

  // Fourth access should evict the oldest
  stats.RecordAccess(400, /*max_k=*/3);
  EXPECT_EQ(stats.recent_access_times_ns.size(), 3u);
  // Most recent is at front
  EXPECT_EQ(stats.recent_access_times_ns[0], 400);
  EXPECT_EQ(stats.recent_access_times_ns[1], 300);
  EXPECT_EQ(stats.recent_access_times_ns[2], 200);
}

TEST_F(ObjectAccessStatsTest, GetKthAccessTimeReturnsCreatedAtForMissingK) {
  ObjectAccessStats stats;
  stats.created_at_ns = 42;
  stats.RecordAccess(100, /*max_k=*/2);
  // K=0 should return the most recent access
  EXPECT_EQ(stats.GetKthAccessTime(0), 100);
  // K=1 should fall back to created_at_ns (only 1 access recorded)
  EXPECT_EQ(stats.GetKthAccessTime(1), 42);
  // Negative k returns created_at_ns
  EXPECT_EQ(stats.GetKthAccessTime(-1), 42);
}

TEST_F(ObjectAccessStatsTest, GetKthAccessTimeWithMultipleAccesses) {
  ObjectAccessStats stats;
  stats.created_at_ns = 0;
  stats.RecordAccess(100, /*max_k=*/3);
  stats.RecordAccess(200, /*max_k=*/3);
  stats.RecordAccess(300, /*max_k=*/3);
  // K=0 → most recent (300)
  EXPECT_EQ(stats.GetKthAccessTime(0), 300);
  // K=1 → 2nd most recent (200)
  EXPECT_EQ(stats.GetKthAccessTime(1), 200);
  // K=2 → 3rd most recent (100)
  EXPECT_EQ(stats.GetKthAccessTime(2), 100);
}

// =====================================================================
// FrequencyWeightedPolicy tests
// =====================================================================

class FrequencyWeightedPolicyTest : public ::testing::Test {
 protected:
  double decay_rate_ = 1e-9;
  FrequencyWeightedPolicy policy_{decay_rate_};
};

TEST_F(FrequencyWeightedPolicyTest, NameIsCorrect) {
  EXPECT_EQ(policy_.Name(), "FrequencyWeighted");
}

TEST_F(FrequencyWeightedPolicyTest, ReturnsEmptyOnZeroBytesToFree) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 0);
  EXPECT_TRUE(result.empty());
}

TEST_F(FrequencyWeightedPolicyTest, ReturnsEmptyOnNegativeBytesToFree) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, -100);
  EXPECT_TRUE(result.empty());
}

TEST_F(FrequencyWeightedPolicyTest, ReturnsEmptyWhenNoPinnedObjects) {
  absl::flat_hash_map<ObjectID, std::unique_ptr<RayObject>> empty_pinned;
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto result = policy_.SelectEvictionCandidates(stats, empty_pinned, AllSpillable, 1000);
  EXPECT_TRUE(result.empty());
}

TEST_F(FrequencyWeightedPolicyTest, EvictsObjectsWithNoStats) {
  // Objects without access stats should have temperature 0 → evicted first.
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // Give one object some access stats (warm)
  ObjectAccessStats warm_stats;
  warm_stats.created_at_ns = 100;
  warm_stats.last_access_ns = 100;
  warm_stats.access_count = 10;
  warm_stats.object_size = 1000;
  stats[ids[1]] = warm_stats;

  // Need to free 1000 bytes → should pick one of the cold objects (ids[0] or ids[2])
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 1000);
  ASSERT_FALSE(result.empty());
  // The evicted object should NOT be the warm one
  EXPECT_NE(result[0], ids[1]);
}

TEST_F(FrequencyWeightedPolicyTest, EvictsColdestObjectFirst) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  int64_t now = absl::GetCurrentTimeNanos();
  // ids[0]: cold (1 access, old)
  stats[ids[0]] = {/*created_at_ns=*/now - 5000000000LL,
                   /*last_access_ns=*/now - 5000000000LL,
                   /*access_count=*/1,
                   /*object_size=*/1000};

  // ids[1]: warm (5 accesses, recent)
  stats[ids[1]] = {/*created_at_ns=*/now - 1000000000LL,
                   /*last_access_ns=*/now - 100000000LL,
                   /*access_count=*/5,
                   /*object_size=*/1000};

  // ids[2]: hot (20 accesses, very recent)
  stats[ids[2]] = {/*created_at_ns=*/now - 500000000LL,
                   /*last_access_ns=*/now,
                   /*access_count=*/20,
                   /*object_size=*/1000};

  // Free 1000 bytes → should evict ids[0] (coldest)
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 1000);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], ids[0]);
}

TEST_F(FrequencyWeightedPolicyTest, FreesEnoughBytes) {
  auto [ids, pinned] = MakePinnedObjects(5, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // All objects have zero stats → all temperature 0 → all equally cold.
  // Need to free 3000 bytes → should select 3 objects.
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 3000);
  EXPECT_EQ(result.size(), 3u);

  // Verify no duplicates
  std::unordered_set<ObjectID> unique_set(result.begin(), result.end());
  EXPECT_EQ(unique_set.size(), result.size());
}

TEST_F(FrequencyWeightedPolicyTest, RespectsSpillableFilter) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // Mark ids[0] and ids[1] as non-spillable
  std::unordered_set<ObjectID> non_spillable{ids[0], ids[1]};
  auto is_spillable = [&non_spillable](const ObjectID &id) {
    return non_spillable.count(id) == 0;
  };

  // Need 3000 bytes but only ids[2] (1000 bytes) is spillable
  auto result = policy_.SelectEvictionCandidates(stats, pinned, is_spillable, 3000);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], ids[2]);
}

TEST_F(FrequencyWeightedPolicyTest, AllObjectsNonSpillableReturnsEmpty) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto none_spillable = [](const ObjectID &) { return false; };

  auto result = policy_.SelectEvictionCandidates(stats, pinned, none_spillable, 1000);
  EXPECT_TRUE(result.empty());
}

TEST_F(FrequencyWeightedPolicyTest, OrdersCorrectlyWithMixedTemperatures) {
  auto [ids, pinned] = MakePinnedObjects(4, 500);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  int64_t now = absl::GetCurrentTimeNanos();

  // Create a clear ordering: ids[3] < ids[0] < ids[2] < ids[1] (temperature)
  stats[ids[0]] = {now - 2000000000LL, now - 2000000000LL, 2, 500};  // lowish
  stats[ids[1]] = {now, now, 100, 500};                              // very hot
  stats[ids[2]] = {now - 100000000LL, now - 100000000LL, 5, 500};    // medium
  stats[ids[3]] = {now - 3000000000LL, now - 3000000000LL, 1, 500};  // coldest

  // Free 1000 bytes (2 objects × 500 bytes)
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 1000);
  ASSERT_EQ(result.size(), 2u);
  // First should be coldest, second should be next coldest
  EXPECT_EQ(result[0], ids[3]);
  EXPECT_EQ(result[1], ids[0]);
}

// =====================================================================
// LRUKPolicy tests
// =====================================================================

class LRUKPolicyTest : public ::testing::Test {
 protected:
  int k_ = 2;
  LRUKPolicy policy_{k_};
};

TEST_F(LRUKPolicyTest, NameIncludesK) {
  EXPECT_EQ(policy_.Name(), "LRU-K(K=2)");
  LRUKPolicy policy3(3);
  EXPECT_EQ(policy3.Name(), "LRU-K(K=3)");
}

TEST_F(LRUKPolicyTest, ReturnsEmptyOnZeroBytesToFree) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 0);
  EXPECT_TRUE(result.empty());
}

TEST_F(LRUKPolicyTest, ReturnsEmptyOnNegativeBytesToFree) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, -500);
  EXPECT_TRUE(result.empty());
}

TEST_F(LRUKPolicyTest, ReturnsEmptyWhenNoPinnedObjects) {
  absl::flat_hash_map<ObjectID, std::unique_ptr<RayObject>> empty_pinned;
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto result = policy_.SelectEvictionCandidates(stats, empty_pinned, AllSpillable, 1000);
  EXPECT_TRUE(result.empty());
}

TEST_F(LRUKPolicyTest, EvictsObjectsWithNoStats) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // Give one object access stats
  ObjectAccessStats obj_stats;
  obj_stats.created_at_ns = 100;
  obj_stats.RecordAccess(200, k_);
  obj_stats.RecordAccess(300, k_);
  stats[ids[1]] = obj_stats;

  // Free 1000 bytes → should pick one without stats (kth_access_time=0)
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 1000);
  ASSERT_FALSE(result.empty());
  EXPECT_NE(result[0], ids[1]);
}

TEST_F(LRUKPolicyTest, EvictsByKthAccessTimeOldestFirst) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // ids[0]: K=2, accesses at 100, 500 → K-th (index 1) = 100
  ObjectAccessStats s0;
  s0.created_at_ns = 50;
  s0.RecordAccess(100, k_);
  s0.RecordAccess(500, k_);
  stats[ids[0]] = s0;

  // ids[1]: K=2, accesses at 200, 600 → K-th (index 1) = 200
  ObjectAccessStats s1;
  s1.created_at_ns = 50;
  s1.RecordAccess(200, k_);
  s1.RecordAccess(600, k_);
  stats[ids[1]] = s1;

  // ids[2]: K=2, accesses at 300, 700 → K-th (index 1) = 300
  ObjectAccessStats s2;
  s2.created_at_ns = 50;
  s2.RecordAccess(300, k_);
  s2.RecordAccess(700, k_);
  stats[ids[2]] = s2;

  // Free 1000 bytes: should evict ids[0] (oldest K-th access = 100)
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 1000);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], ids[0]);
}

TEST_F(LRUKPolicyTest, FallsBackToCreatedAtWhenFewerThanKAccesses) {
  auto [ids, pinned] = MakePinnedObjects(2, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // ids[0]: only 1 access → K-th (k-1 = 1) falls back to created_at_ns = 50
  ObjectAccessStats s0;
  s0.created_at_ns = 50;
  s0.RecordAccess(900, k_);
  stats[ids[0]] = s0;

  // ids[1]: 2 accesses → K-th (index 1) = 200
  ObjectAccessStats s1;
  s1.created_at_ns = 100;
  s1.RecordAccess(200, k_);
  s1.RecordAccess(800, k_);
  stats[ids[1]] = s1;

  // Free 1000 bytes: ids[0] should be evicted (created_at=50 < Kth=200)
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 1000);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], ids[0]);
}

TEST_F(LRUKPolicyTest, FreesEnoughBytes) {
  auto [ids, pinned] = MakePinnedObjects(5, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // All have kth_access = 0 (no stats) → any 3 will do
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 3000);
  EXPECT_EQ(result.size(), 3u);
  std::unordered_set<ObjectID> unique_set(result.begin(), result.end());
  EXPECT_EQ(unique_set.size(), result.size());
}

TEST_F(LRUKPolicyTest, RespectsSpillableFilter) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  std::unordered_set<ObjectID> non_spillable{ids[0], ids[2]};
  auto is_spillable = [&non_spillable](const ObjectID &id) {
    return non_spillable.count(id) == 0;
  };

  auto result = policy_.SelectEvictionCandidates(stats, pinned, is_spillable, 3000);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0], ids[1]);
}

TEST_F(LRUKPolicyTest, AllObjectsNonSpillableReturnsEmpty) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;
  auto none_spillable = [](const ObjectID &) { return false; };

  auto result = policy_.SelectEvictionCandidates(stats, pinned, none_spillable, 1000);
  EXPECT_TRUE(result.empty());
}

TEST_F(LRUKPolicyTest, OrdersCorrectlyWithK3) {
  LRUKPolicy policy3(3);
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // ids[0]: 3 accesses at 100, 200, 300 → K-th (index 2) = 100
  ObjectAccessStats s0;
  s0.created_at_ns = 0;
  s0.RecordAccess(100, 3);
  s0.RecordAccess(200, 3);
  s0.RecordAccess(300, 3);
  stats[ids[0]] = s0;

  // ids[1]: 3 accesses at 50, 150, 250 → K-th (index 2) = 50
  ObjectAccessStats s1;
  s1.created_at_ns = 0;
  s1.RecordAccess(50, 3);
  s1.RecordAccess(150, 3);
  s1.RecordAccess(250, 3);
  stats[ids[1]] = s1;

  // ids[2]: 3 accesses at 400, 500, 600 → K-th (index 2) = 400
  ObjectAccessStats s2;
  s2.created_at_ns = 0;
  s2.RecordAccess(400, 3);
  s2.RecordAccess(500, 3);
  s2.RecordAccess(600, 3);
  stats[ids[2]] = s2;

  // Free 2000 bytes → 2 objects. Eviction order: ids[1] (50), ids[0] (100)
  auto result = policy3.SelectEvictionCandidates(stats, pinned, AllSpillable, 2000);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], ids[1]);
  EXPECT_EQ(result[1], ids[0]);
}

TEST_F(LRUKPolicyTest, EvictsAllIfBytesExceedTotal) {
  auto [ids, pinned] = MakePinnedObjects(3, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // Request more bytes than available → should return all spillable objects
  auto result = policy_.SelectEvictionCandidates(stats, pinned, AllSpillable, 100000);
  EXPECT_EQ(result.size(), 3u);
}

// =====================================================================
// Cross-policy comparison tests
// =====================================================================

class PolicyComparisonTest : public ::testing::Test {};

TEST_F(PolicyComparisonTest, BothPoliciesReturnSubsetOfSpillable) {
  auto [ids, pinned] = MakePinnedObjects(5, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // Mark some as non-spillable
  std::unordered_set<ObjectID> non_spillable{ids[0], ids[3]};
  auto is_spillable = [&non_spillable](const ObjectID &id) {
    return non_spillable.count(id) == 0;
  };

  FrequencyWeightedPolicy fw(1e-9);
  LRUKPolicy lruk(2);

  auto fw_result = fw.SelectEvictionCandidates(stats, pinned, is_spillable, 2000);
  auto lruk_result = lruk.SelectEvictionCandidates(stats, pinned, is_spillable, 2000);

  for (const auto &id : fw_result) {
    EXPECT_EQ(non_spillable.count(id), 0u)
        << "FrequencyWeighted returned non-spillable object";
  }
  for (const auto &id : lruk_result) {
    EXPECT_EQ(non_spillable.count(id), 0u) << "LRU-K returned non-spillable object";
  }
}

TEST_F(PolicyComparisonTest, BothPoliciesReturnOnlyPinnedObjects) {
  auto [ids, pinned] = MakePinnedObjects(4, 1000);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  // Add stats for an object NOT in pinned_objects
  ObjectID phantom = ObjectID::FromRandom();
  stats[phantom] = {0, 0, 1, 1000};

  FrequencyWeightedPolicy fw(1e-9);
  LRUKPolicy lruk(2);

  auto fw_result = fw.SelectEvictionCandidates(stats, pinned, AllSpillable, 2000);
  auto lruk_result = lruk.SelectEvictionCandidates(stats, pinned, AllSpillable, 2000);

  for (const auto &id : fw_result) {
    EXPECT_TRUE(pinned.count(id) > 0)
        << "FrequencyWeighted returned object not in pinned_objects";
  }
  for (const auto &id : lruk_result) {
    EXPECT_TRUE(pinned.count(id) > 0) << "LRU-K returned object not in pinned_objects";
  }
}

TEST_F(PolicyComparisonTest, NoDuplicatesInResults) {
  auto [ids, pinned] = MakePinnedObjects(10, 100);
  absl::flat_hash_map<ObjectID, ObjectAccessStats> stats;

  FrequencyWeightedPolicy fw(1e-9);
  LRUKPolicy lruk(2);

  auto fw_result = fw.SelectEvictionCandidates(stats, pinned, AllSpillable, 500);
  auto lruk_result = lruk.SelectEvictionCandidates(stats, pinned, AllSpillable, 500);

  std::unordered_set<ObjectID> fw_set(fw_result.begin(), fw_result.end());
  EXPECT_EQ(fw_set.size(), fw_result.size()) << "FrequencyWeighted has duplicates";

  std::unordered_set<ObjectID> lruk_set(lruk_result.begin(), lruk_result.end());
  EXPECT_EQ(lruk_set.size(), lruk_result.size()) << "LRU-K has duplicates";
}

}  // namespace raylet
}  // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
