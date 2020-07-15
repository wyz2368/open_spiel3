// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/algorithms/corr_dev_builder.h"

#include "open_spiel/policy.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace algorithms {

CorrDevBuilder::CorrDevBuilder(int seed) : rng_(seed), total_weight_(0.0) {}

void CorrDevBuilder::AddDeterminsticJointPolicy(const TabularPolicy& policy,
                                                double weight) {
  std::string key = policy.ToStringSorted();
  auto iter = policy_weights_.find(key);
  if (iter == policy_weights_.end()) {
    policy_weights_[key] = weight;
    policy_map_[key] = policy;
  } else {
    iter->second += weight;
  }
  total_weight_ += weight;
}

void CorrDevBuilder::AddSampledJointPolicy(const TabularPolicy& policy,
                                           int num_samples) {
  for (int sample = 0; sample < num_samples; ++sample) {
    TabularPolicy sampled_policy;
    for (const auto& iter : policy.PolicyTable()) {
      Action sampled_action = SampleAction(iter.second, rng_).first;
      sampled_policy.SetStatePolicy(
          iter.first, ToDeterministicPolicy(iter.second, sampled_action));
    }
    AddDeterminsticJointPolicy(sampled_policy, 1.0 / num_samples);
  }
}

CorrelationDevice CorrDevBuilder::GetCorrelationDevice() const {
  SPIEL_CHECK_GT(total_weight_, 0);
  CorrelationDevice corr_dev;
  double sum_weight = 0;
  for (const auto& key_and_policy : policy_map_) {
    double weight = policy_weights_.at(key_and_policy.first);
    sum_weight += weight;
    corr_dev.push_back({weight / total_weight_, key_and_policy.second});
  }
  SPIEL_CHECK_TRUE(Near(sum_weight, total_weight_));
  return corr_dev;
}

}  // namespace algorithms
}  // namespace open_spiel
