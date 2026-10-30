#include "llama-moe-profile.h"
#include "llama-model.h"
#include "llama-ext.h"

#include "../vendor/nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
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
    std::vector<uint64_t> seed;
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
            if (model.hparams.no_alloc) { continue; }
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

namespace {
struct profile_source {
    std::string file;
    std::filesystem::path lock;
    json data = nullptr;
    std::string bytes;
    bool exists = false, locked = false;
    std::atomic<bool> writer{false};

    ~profile_source() {
        if (!locked) { return; }
        std::error_code error;
        std::filesystem::remove(lock/"profile.tmp", error);
        std::filesystem::remove(lock, error);
    }
};

std::string read_profile(const std::string & path, bool & exists) {
    if (std::filesystem::is_symlink(path)) { throw std::runtime_error("MoE route profile: symlink rejected: " + path); }
    exists = std::filesystem::exists(path);
    if (!exists) { return {}; }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    const auto size = input.tellg();
    if (!input || size <= 0 || size > std::streamoff(max_file_bytes)) {
        throw std::runtime_error("MoE route profile: invalid file size: " + path);
    }
    std::string bytes(size_t(size), '\0');
    input.seekg(0); input.read(bytes.data(), bytes.size());
    if (!input) { throw std::runtime_error("MoE route profile: read failed: " + path); }
    return bytes;
}
} // namespace

struct llama_moe_profile_snapshot {
    std::shared_ptr<profile_source> source;
};

std::shared_ptr<const llama_moe_profile_snapshot> llama_moe_profile_snapshot_load(const char * path) {
    if (!path || !path[0]) { throw std::invalid_argument("MoE route profile: empty snapshot path"); }
    auto snapshot = std::make_shared<llama_moe_profile_snapshot>();
    auto & state = snapshot->source = std::make_shared<profile_source>();
    state->file = path;
    state->lock = std::filesystem::path(std::string(path) + ".lock");
    if (!std::filesystem::create_directory(state->lock)) { throw std::runtime_error("MoE route profile: file is locked: " + state->file); }
    state->locked = true;
    state->bytes = read_profile(path, state->exists);
    if (state->exists) {
        state->data = json::parse(state->bytes);
        const auto checksum = state->data.at("checksum").get<std::string>();
        state->data.erase("checksum");
        if (checksum != hash_string(state->data.dump())) { throw std::runtime_error("MoE route profile: checksum mismatch: " + state->file); }
    }
    return snapshot;
}

struct llama_moe_route_profile::impl {
    std::string file;
    std::shared_ptr<profile_source> source;
    std::string expected_bytes;
    bool expected_exists = false, writer = false, metadata = false;
    json identity;
    json seed_provenance = nullptr;
    std::map<int, counts> layers;

    ~impl() {
        if (writer) { source->writer.store(false); }
    }

    json document() const {
        json rows = json::array();
        for (const auto & entry : layers) {
            json route_counts, call_counts;
            for (size_t i = 0; i < phases.size(); ++i) {
                route_counts[phases[i]] = entry.second.routes[i];
                call_counts[phases[i]] = entry.second.calls[i];
            }
            rows.push_back({{"layer", entry.first}, {"counts", route_counts}, {"calls", call_counts}, {"seed_counts", entry.second.seed}});
        }
        return {{"format", "llama-moe-route-profile"}, {"version", 2}, {"namespace", "target"},
            {"seed_provenance", seed_provenance},
            {"identity", identity}, {"coverage", "cached host layers; short cache ops and staged prefill only"},
            {"semantics", "executed routes, including rejected verification positions; no accepted-token filtering"},
            {"phases", "logical decode-call width:1 single_token,2..32 short_batch,>32 prefill"},
            {"layers", rows}};
    }
};

llama_moe_route_profile::llama_moe_route_profile(const llama_model & model, const std::string & path,
        const llama_moe_profile_snapshot * snapshot) : pimpl(new impl) {
    const auto owned = snapshot ? nullptr : llama_moe_profile_snapshot_load(path.c_str());
    if (!snapshot) { snapshot = owned.get(); }
    auto & state = *pimpl;
    state.file = path;
    state.source = snapshot->source;
    if (!state.source || state.source->file != path) { throw std::invalid_argument("MoE route profile: snapshot path mismatch"); }
    state.metadata = model.hparams.no_alloc;
    if (!state.metadata) {
        if (state.source->writer.exchange(true)) { throw std::runtime_error("MoE route profile: snapshot already has a writer"); }
        state.writer = true;
    }
    state.expected_bytes = state.source->bytes;
    state.expected_exists = state.source->exists;
    state.identity = model_identity(model);
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & layer = model.layers[il];
        if (!layer.ffn_up_exps || !layer.ffn_gate_exps || !layer.ffn_down_exps) { continue; }
        auto & row = state.layers[int(il)];
        for (auto & values : row.routes) { values.resize(layer.ffn_up_exps->ne[2]); }
        row.seed.resize(layer.ffn_up_exps->ne[2]);
    }
    if (!state.source->exists) { return; }
    const auto & data = state.source->data;
    auto expected = state.document();
    auto actual_identity = data.at("identity");
    if (state.metadata) {
        // Metadata-fit validates layout and counts without touching weight payloads.
        actual_identity.erase("weight_sample_fnv64");
        expected["identity"].erase("weight_sample_fnv64");
    }
    if (!data.at("version").is_number_unsigned() ||
        data.at("format") != expected["format"] ||
        (data.at("version") != 1 && data.at("version") != 2) || data.at("namespace") != expected["namespace"] ||
        actual_identity != expected["identity"] || data.at("coverage") != expected["coverage"] ||
        data.at("semantics") != expected["semantics"] || data.at("phases") != expected["phases"] ||
        !data.at("layers").is_array() || data["layers"].size() != state.layers.size()) {
        throw std::runtime_error("MoE route profile: checksum, schema or model identity mismatch: " + path);
    }
    state.identity = data.at("identity");
    const bool seeded_format = data["version"] == 2;
    if (seeded_format) {
        state.seed_provenance = data.at("seed_provenance");
        if (!state.seed_provenance.is_null()) {
            const auto & origin = state.seed_provenance;
            auto valid_hash = [](const json & value) {
                if (!value.is_string()) { return false; }
                const auto str = value.get<std::string>();
                return str.size() == 64 && str.find_first_not_of("0123456789abcdef") == std::string::npos;
            };
            if (!origin.is_object() || origin.at("kind") != "maya" ||
                !valid_hash(origin.at("source_sha256")) || !valid_hash(origin.at("template_sha256")) ||
                !origin.at("target_layers").is_number_unsigned() ||
                origin.at("target_layers") != state.layers.size() ||
                !origin.at("profile_rows").is_number_unsigned() ||
                origin["profile_rows"].get<uint64_t>() < state.layers.size() ||
                origin["profile_rows"].get<uint64_t>() > state.layers.size() + 3) {
                throw std::runtime_error("MoE route profile: invalid seed provenance");
            }
        }
    } else if (data.contains("seed_provenance")) {
        throw std::runtime_error("MoE route profile: seed requires format version 2");
    }
    size_t index = 0;
    for (auto & entry : state.layers) {
        const auto & row = data["layers"][index++];
        if (row.at("layer") != entry.first) { throw std::runtime_error("MoE route profile: invalid layer ID"); }
        if (seeded_format) {
            const auto & seed = row.at("seed_counts");
            if (!seed.is_array() || seed.size() != entry.second.seed.size()) {
                throw std::runtime_error("MoE route profile: invalid seed shape");
            }
            for (size_t e = 0; e < seed.size(); ++e) {
                if (!seed[e].is_number_unsigned()) { throw std::runtime_error("MoE route profile: invalid seed count"); }
                entry.second.seed[e] = seed[e].get<uint64_t>();
                if (entry.second.seed[e] && state.seed_provenance.is_null()) {
                    throw std::runtime_error("MoE route profile: nonempty seed requires provenance");
                }
            }
        } else if (row.contains("seed_counts")) {
            throw std::runtime_error("MoE route profile: seed requires format version 2");
        }
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
    if (pimpl->metadata) { throw std::logic_error("MoE route profile: metadata snapshot is read only"); }
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
    if (pimpl->metadata) { throw std::logic_error("MoE route profile: metadata snapshot is read only"); }
    bool exists;
    if (read_profile(pimpl->file, exists) != pimpl->expected_bytes || exists != pimpl->expected_exists) {
        throw std::runtime_error("MoE route profile: source changed after snapshot; file preserved");
    }
    auto data = pimpl->document();
    data["checksum"] = hash_string(data.dump());
    const auto serialized = data.dump(2) + '\n';
    const auto temp = pimpl->source->lock/"profile.tmp";
    std::ofstream output(temp, std::ios::binary | std::ios::trunc);
    output << serialized;
    output.flush();
    if (!output) { throw std::runtime_error("MoE route profile: write failed: " + pimpl->file); }
    output.close();
    if (!output) { throw std::runtime_error("MoE route profile: close failed: " + pimpl->file); }
    std::filesystem::rename(temp, pimpl->file);
    pimpl->expected_bytes = serialized;
    pimpl->expected_exists = true;
}

const std::string & llama_moe_route_profile::path() const {
    return pimpl->file;
}


std::vector<int32_t> llama_moe_route_profile::hottest(int layer, int limit) const {
    const auto found = pimpl->layers.find(layer);
    if (found == pimpl->layers.end() || limit <= 0) { return {}; }
    const auto & row = found->second;
    auto add = [](uint64_t a, uint64_t b) { return a + std::min(b, UINT64_MAX - a); };
    std::vector<uint64_t> heat(row.seed.size());
    for (size_t e = 0; e < heat.size(); ++e) {
        heat[e] = add(row.seed[e], add(row.routes[0][e], row.routes[1][e]));
    }
    if (std::all_of(heat.begin(), heat.end(), [](uint64_t v) { return v == 0; })) {
        heat = row.routes[2];
    }
    std::vector<int32_t> ids;
    for (size_t e = 0; e < heat.size(); ++e) {
        if (heat[e]) { ids.push_back(int32_t(e)); }
    }
    std::sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) {
        return heat[a] != heat[b] ? heat[a] > heat[b] : a < b;
    });
    if (ids.size() > size_t(limit)) { ids.resize(limit); }
    return ids;
}
