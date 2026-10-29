#include "llama-moe-profile.h"
#include "llama-model.h"

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using json = nlohmann::json;
constexpr size_t max_file_bytes = 16*1024*1024;
const std::array<const char *, 3> phases = {"single_token", "short_batch", "prefill"};

uint64_t hash_bytes(const void * data, size_t size, uint64_t hash = UINT64_C(14695981039346656037)) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) { hash = (hash ^ bytes[i])*UINT64_C(1099511628211); }
    return hash;
}

std::string hash_string(const std::string & value) {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash_bytes(value.data(), value.size());
    return out.str();
}

void increment(uint64_t & value) {
    if (value != std::numeric_limits<uint64_t>::max()) { ++value; }
}

struct counts {
    std::array<std::vector<uint64_t>, 3> routes;
    std::array<uint64_t, 3> calls{};
};

json model_identity(const llama_model & model) {
    std::map<std::string, std::string> metadata(model.gguf_kv.begin(), model.gguf_kv.end());
    json layout = json::array();
    uint64_t samples = UINT64_C(14695981039346656037);
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        if (!layer.ffn_up_exps || !layer.ffn_gate_exps || !layer.ffn_down_exps) { continue; }
        json tensors = json::array();
        for (const auto * tensor : {layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps}) {
            tensors.push_back({{"name", tensor->name}, {"type", ggml_type_name(tensor->type)},
                {"shape", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}}});
            std::array<uint8_t, 128> sample{};
            const size_t size = std::min(sample.size(), ggml_nbytes(tensor));
            for (size_t offset : {size_t(0), ggml_nbytes(tensor) - size}) {
                ggml_backend_tensor_get(tensor, sample.data(), offset, size);
                samples = hash_bytes(sample.data(), size, samples);
            }
        }
        layout.push_back({{"layer", il}, {"tensors", tensors}});
    }
    std::ostringstream sample_hash;
    sample_hash << std::hex << std::setfill('0') << std::setw(16) << samples;
    return {{"architecture", model.arch_name()}, {"metadata_fnv64", hash_string(json(metadata).dump())},
        {"layout", layout}, {"weight_sample_fnv64", sample_hash.str()},
        {"weight_sample_scope", "first/last128 bytes of each routed projection; not a full weight checksum"}};
}
} // namespace

struct llama_moe_route_profile::impl {
    std::string file;
    std::filesystem::path lock;
    json identity;
    std::map<int, counts> layers;
    bool locked = false;

    ~impl() {
        if (locked) {
            std::error_code error;
            std::filesystem::remove(lock/"profile.tmp", error);
            std::filesystem::remove(lock, error);
        }
    }

    json document() const {
        json rows = json::array();
        for (const auto & entry : layers) {
            json route_counts, call_counts;
            for (size_t i = 0; i < phases.size(); ++i) {
                route_counts[phases[i]] = entry.second.routes[i];
                call_counts[phases[i]] = entry.second.calls[i];
            }
            rows.push_back({{"layer", entry.first}, {"counts", route_counts}, {"calls", call_counts}});
        }
        return {{"format", "llama-moe-route-profile"}, {"version", 1}, {"namespace", "target"},
            {"identity", identity}, {"coverage", "cached host layers; short cache ops and staged prefill only"},
            {"semantics", "executed routes, including rejected verification positions; no accepted-token filtering"},
            {"phases", "logical decode-call width:1 single_token,2..32 short_batch,>32 prefill"},
            {"layers", rows}};
    }
};

llama_moe_route_profile::llama_moe_route_profile(const llama_model & model, const std::string & path) : pimpl(new impl) {
    auto & state = *pimpl;
    state.file = path;
    state.lock = std::filesystem::path(path + ".lock");
    if (!std::filesystem::create_directory(state.lock)) {
        throw std::runtime_error("MoE route profile: file is locked: " + path);
    }
    state.locked = true;
    state.identity = model_identity(model);
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        if (!layer.ffn_up_exps || !layer.ffn_gate_exps || !layer.ffn_down_exps) { continue; }
        auto & row = state.layers[int(il)];
        for (auto & values : row.routes) { values.resize(layer.ffn_up_exps->ne[2]); }
    }
    if (!std::filesystem::exists(path)) { return; }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto size = input.tellg();
    if (!input || size <= 0 || size > std::streamoff(max_file_bytes)) {
        throw std::runtime_error("MoE route profile: invalid file size: " + path);
    }
    input.seekg(0);
    json data = json::parse(input);
    const auto checksum = data.at("checksum").get<std::string>();
    data.erase("checksum");
    auto expected = state.document();
    if (!data.at("version").is_number_unsigned() || checksum != hash_string(data.dump()) ||
        data.at("format") != expected["format"] ||
        data.at("version") != expected["version"] || data.at("namespace") != expected["namespace"] ||
        data.at("identity") != expected["identity"] || data.at("coverage") != expected["coverage"] ||
        data.at("semantics") != expected["semantics"] || data.at("phases") != expected["phases"] ||
        !data.at("layers").is_array() || data["layers"].size() != state.layers.size()) {
        throw std::runtime_error("MoE route profile: checksum, schema or model identity mismatch: " + path);
    }
    size_t index = 0;
    for (auto & entry : state.layers) {
        const auto & row = data["layers"][index++];
        if (row.at("layer") != entry.first) { throw std::runtime_error("MoE route profile: invalid layer ID"); }
        for (size_t i = 0; i < phases.size(); ++i) {
            const auto & values = row.at("counts").at(phases[i]);
            const auto & calls = row.at("calls").at(phases[i]);
            if (!values.is_array() || values.size() != entry.second.routes[i].size() || !calls.is_number_unsigned()) {
                throw std::runtime_error("MoE route profile: invalid counter shape or type");
            }
            for (size_t e = 0; e < values.size(); ++e) {
                if (!values[e].is_number_unsigned()) { throw std::runtime_error("MoE route profile: invalid route count"); }
                entry.second.routes[i][e] = values[e].get<uint64_t>();
            }
            entry.second.calls[i] = calls.get<uint64_t>();
        }
    }
}

llama_moe_route_profile::~llama_moe_route_profile() = default;

void llama_moe_route_profile::observe(int layer, int phase, const ggml_tensor * ids) {
    const auto found = pimpl->layers.find(layer);
    if (found == pimpl->layers.end() || phase < 0 || phase >= int(phases.size())) { return; }
    auto & values = found->second.routes[phase];
    increment(found->second.calls[phase]);
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t k = 0; k < ids->ne[0]; ++k) {
            int32_t id;
            std::memcpy(&id, static_cast<const char *>(ids->data) + t*ids->nb[1] + k*ids->nb[0], sizeof(id));
            if (id >= 0 && size_t(id) < values.size()) { increment(values[id]); }
        }
    }
}

void llama_moe_route_profile::save() {
    auto data = pimpl->document();
    data["checksum"] = hash_string(data.dump());
    const auto temp = pimpl->lock/"profile.tmp";
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    output << data.dump(2) << '\n';
    output.flush();
    if (!output) { throw std::runtime_error("MoE route profile: write failed: " + pimpl->file); }
    output.close();
    if (!output) { throw std::runtime_error("MoE route profile: close failed: " + pimpl->file); }
    std::filesystem::rename(temp, pimpl->file);
}

const std::string & llama_moe_route_profile::path() const {
    return pimpl->file;
}
