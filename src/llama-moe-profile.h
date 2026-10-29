#pragma once

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

struct llama_model;
struct ggml_tensor;

// Counts executed routes at existing cache observation points.
struct llama_moe_route_profile {
    llama_moe_route_profile(const llama_model & model, const std::string & path);
    ~llama_moe_route_profile();

    void observe(int layer, int phase, const ggml_tensor * ids);
    void save();
    std::vector<int32_t> hottest(int layer, int limit) const;
    const std::string & path() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
