#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

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
