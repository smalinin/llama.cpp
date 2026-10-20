#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Each denoising block length has its own acceptance history.
class common_speculative_adaptive_verify {
public:
    void reset(const std::vector<double> & costs = {}) {
        for (double cost : costs) {
            if (!std::isfinite(cost) || cost <= 0.0) {
                throw std::invalid_argument("adaptive verification costs must be finite and positive");
            }
        }
        costs_ = costs;
        survival_.assign(costs.size(), {});
        samples_.assign(costs.size(), 0);
        for (size_t k = 0; k < costs.size(); ++k) {
            survival_[k].assign(k, 0.0);
        }
        previous_k_ = 0;
        until_probe_ = 0;
        probe_k_ = 0;
        exploring_ = false;
    }

    bool empty() const { return costs_.empty(); }
    bool exploring() const { return exploring_; }
    uint64_t samples(int32_t k) const { return samples_.at(k); }
    double cost(int32_t k) const { return costs_.at(k); }
    double survival(int32_t k, int32_t i) const { return survival_.at(k).at(i); }

    // Call before draft forward; short verification uses at most two draft tokens.
    int32_t select(int32_t hard_max, int32_t min_observations, int32_t probe_interval,
                   double safety_margin, double hysteresis) {
        if (empty() || min_observations < 1 || probe_interval < 1 ||
                !std::isfinite(safety_margin) || safety_margin < 0.0 || safety_margin >= 1.0 ||
                !std::isfinite(hysteresis) || hysteresis < 0.0) {
            throw std::invalid_argument("invalid adaptive verification parameters");
        }
        const int32_t n = std::clamp(hard_max, 0, std::min<int32_t>(2, costs_.size() - 1));
        exploring_ = true;
        for (int32_t k = 0; k <= n; ++k) {
            if (samples_[k] < (uint64_t) min_observations) {
                return k;
            }
        }

        auto goodput = [&](int32_t k) {
            double tokens = 1.0;
            for (double rate : survival_[k]) {
                tokens += std::max(0.0, rate - safety_margin);
            }
            return tokens / costs_[k];
        };
        int32_t best = 0;
        for (int32_t k = 1; k <= n; ++k) {
            if (goodput(k) > goodput(best)) {
                best = k;
            }
        }
        previous_k_ = std::min(previous_k_, n);
        if (goodput(best) <= goodput(previous_k_) * (1.0 + hysteresis)) {
            best = previous_k_;
        }
        previous_k_ = best;

        if (++until_probe_ >= probe_interval && n > 0) {
            until_probe_ = 0;
            do {
                probe_k_ = (probe_k_ + 1) % (n + 1);
            } while (probe_k_ == best);
            return probe_k_;
        }
        exploring_ = false;
        return best;
    }

    // Ignore truncated blocks: their acceptance is not a sample of this shape.
    void observe(int32_t requested, int32_t verified, int32_t accepted, double cost_us, double alpha) {
        if (requested < 0 || requested >= (int32_t) costs_.size() || requested != verified ||
                accepted < 0 || accepted > verified || !std::isfinite(cost_us) || cost_us <= 0.0 ||
                !std::isfinite(alpha) || alpha <= 0.0 || alpha > 1.0) {
            return;
        }
        auto & rates = survival_[requested];
        for (int32_t i = 0; i < verified; ++i) {
            const double rate = accepted > i ? 1.0 : 0.0;
            rates[i] = samples_[requested] == 0 ? rate : alpha * rate + (1.0 - alpha) * rates[i];
        }
        costs_[requested] = alpha * cost_us + (1.0 - alpha) * costs_[requested];
        samples_[requested]++;
    }

private:
    std::vector<double> costs_;
    std::vector<std::vector<double>> survival_;
    std::vector<uint64_t> samples_;
    int32_t previous_k_ = 0;
    int32_t until_probe_ = 0;
    int32_t probe_k_ = 0;
    bool exploring_ = false;
};

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
