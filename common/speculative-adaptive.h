#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

struct common_speculative_adaptive_verify_result {
    int32_t k = 0;
    std::vector<double> survival;
    std::vector<double> expected_tokens;
    std::vector<double> goodput;
};

// Select the target verification prefix that maximizes expected output tokens
// per unit cost.  Confidence values are interpreted as conditional acceptance
// probabilities, so the probability of reaching position i is their cumulative
// product.  Costs are supplied by the caller and contain entries for K=0..N;
// this deliberately keeps hardware- and quant-specific timings out of common.
inline common_speculative_adaptive_verify_result common_speculative_adaptive_verify_select(
        const std::vector<float>  & confidence,
        const std::vector<double> & costs,
        int32_t previous_k,
        int32_t min_k,
        double safety_margin,
        double hysteresis) {
    if (costs.empty()) {
        throw std::invalid_argument("adaptive verification needs at least the K=0 cost");
    }
    for (double cost : costs) {
        if (!std::isfinite(cost) || cost <= 0.0) {
            throw std::invalid_argument("adaptive verification costs must be finite and positive");
        }
    }
    if (!std::isfinite(safety_margin) || safety_margin < 0.0 || safety_margin >= 1.0) {
        throw std::invalid_argument("adaptive verification safety margin must be in [0, 1)");
    }
    if (!std::isfinite(hysteresis) || hysteresis < 0.0) {
        throw std::invalid_argument("adaptive verification hysteresis must be finite and non-negative");
    }

    const int32_t n = std::min<int32_t>(confidence.size(), costs.size() - 1);
    min_k = std::clamp(min_k, 0, n);

    common_speculative_adaptive_verify_result result;
    result.survival.resize(n);
    result.expected_tokens.resize(n + 1, 1.0);
    result.goodput.resize(n + 1);
    result.goodput[0] = 1.0 / costs[0];

    double survival = 1.0;
    for (int32_t i = 0; i < n; ++i) {
        const double p = std::clamp((double) confidence[i] - safety_margin, 0.0, 1.0);
        survival *= p;
        result.survival[i] = survival;
        result.expected_tokens[i + 1] = result.expected_tokens[i] + survival;
        result.goodput[i + 1] = result.expected_tokens[i + 1] / costs[i + 1];
    }

    int32_t best_k = min_k;
    for (int32_t k = min_k + 1; k <= n; ++k) {
        // Strict comparison makes ties deterministic and favors the cheaper,
        // smaller graph shape.
        if (result.goodput[k] > result.goodput[best_k]) {
            best_k = k;
        }
    }

    if (previous_k >= min_k && previous_k <= n &&
            result.goodput[best_k] <= result.goodput[previous_k] * (1.0 + hysteresis)) {
        best_k = previous_k;
    }

    result.k = best_k;
    return result;
}

// Adjust the MTP draft length from per-position acceptance history.
class common_speculative_adaptive_draft {
public:
    common_speculative_adaptive_draft(int32_t n_min, int32_t n_max, bool enabled = true)
        : n_min_(n_max > 0 ? std::clamp(std::max(1, n_min), 1, n_max) : 0),
          n_max_(std::max(0, n_max)),
          enabled_(enabled && n_max_ > n_min_),
          rates_(n_max_, 1.0f),
          samples_(n_max_, 0) {
        reset();
    }

    void reset() {
        n_cur_ = n_max_;
        n_last_ = 0;
        n_full_ = 0;
        std::fill(rates_.begin(), rates_.end(), 1.0f);
        std::fill(samples_.begin(), samples_.end(), 0);
    }

    int32_t limit(int32_t hard_max) const {
        return std::max(0, std::min(n_cur_, hard_max));
    }

    void drafted(int32_t n_drafted) {
        n_last_ = std::clamp(n_drafted, 0, n_max_);
    }

    bool accept(int32_t n_accepted) {
        if (!enabled_ || n_last_ <= 0) {
            n_last_ = 0;
            return false;
        }

        n_accepted = std::clamp(n_accepted, 0, n_last_);
        for (int32_t i = 0; i < n_last_; ++i) {
            const float observed = n_accepted > i ? 1.0f : 0.0f;
            rates_[i] = samples_[i] == 0 ? observed : 0.75f*rates_[i] + 0.25f*observed;
            samples_[i]++;
        }

        const int32_t old = n_cur_;

        // Do not learn from a draft shortened by p_min or the remaining context.
        if (n_last_ == n_cur_) {
            if (n_accepted == n_last_) {
                n_full_++;
                if (n_full_ >= 4 && n_cur_ < n_max_) {
                    rates_[n_cur_] = 1.0f;
                    samples_[n_cur_] = 0;
                    n_cur_++;
                    n_full_ = 0;
                }
            } else {
                n_full_ = 0;
                const int32_t pos = n_cur_ - 1;
                if (n_cur_ > n_min_ && samples_[pos] >= 4 && rates_[pos] < 0.55f) {
                    n_cur_--;
                }
            }
        } else {
            n_full_ = 0;
        }

        n_last_ = 0;
        return old != n_cur_;
    }

    int32_t current() const {
        return n_cur_;
    }

    float rate(int32_t pos) const {
        return pos >= 0 && pos < n_max_ ? rates_[pos] : 0.0f;
    }

private:
    int32_t n_min_ = 0;
    int32_t n_max_ = 0;
    bool enabled_ = false;

    int32_t n_cur_ = 0;
    int32_t n_last_ = 0;
    int32_t n_full_ = 0;

    std::vector<float> rates_;
    std::vector<int32_t> samples_;
};
