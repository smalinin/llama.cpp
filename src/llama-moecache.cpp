#include "llama-moecache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)

    std::vector<bool>     slot_in_flight; // slot has an upload pending
    std::vector<bool>     expert_in_flight;

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;
    uint64_t n_upload = 0;
    uint64_t n_published = 0;
    uint64_t n_failed = 0;
    uint64_t n_evicted = 0;
    uint64_t upload_bytes = 0;
    int64_t upload_us = 0;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    bool    success = false;
};

struct moe_cache {
    const llama_model * model = nullptr;

    int32_t n_slots     = 0;
    int32_t max_inserts = 2;
    uint64_t budget_bytes = 0;
    ggml_backend_dev_t budget_device = nullptr;

    uint64_t clock   = 0;
    uint64_t n_steps = 0;
    bool profile = false;
    int64_t sync_wait_us = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;
    std::map<const ggml_tensor *, size_t> by_observe_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;
    std::map<ggml_backend_buffer_type_t, size_t> expected_bytes;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    std::vector<upload_job>  done;
    bool                     stop = false;
};

moe_cache * g_cache = nullptr;
std::map<const llama_model *, moe_cache *> g_metadata;
std::mutex g_init_mtx;
bool g_init_done = false;

bool host_projection(const ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_is_host(t->buffer);
}

ggml_backend_buffer_type_t cache_home(const llama_layer & l) {
    for (const auto * t : {l.ffn_gate_inp, l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps}) {
        if (t && t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            auto buft = ggml_backend_buffer_get_type(t->buffer);
            if (ggml_backend_buft_get_device(buft)) {
                return buft;
            }
        }
    }
    return nullptr;
}

bool cacheable_layer(const llama_layer & l) {
    if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
        return false;
    }
    // Float MUL_MAT_ID kernels need distinct ids; cache misses share the dummy slot.
    if (!ggml_is_quantized(l.ffn_up_exps->type) || !ggml_is_quantized(l.ffn_gate_exps->type) || !ggml_is_quantized(l.ffn_down_exps->type)) {
        return false;
    }
    if (!l.ffn_up_exps->buffer || !l.ffn_gate_exps->buffer || !l.ffn_down_exps->buffer ||
        (!host_projection(l.ffn_up_exps) && !host_projection(l.ffn_gate_exps) && !host_projection(l.ffn_down_exps))) {
        return false;
    }
    for (const auto * t : {l.ffn_up_exps, l.ffn_gate_exps, l.ffn_down_exps}) {
        if ((host_projection(t) && t->extra) || !ggml_is_contiguous(t) || t->ne[2] != l.ffn_up_exps->ne[2] || t->ne[3] != 1) {
            return false;
        }
    }
    return cache_home(l) != nullptr;
}

struct stage_sizes {
    ggml_backend_buffer_type_t host_type = nullptr;
    size_t weights = 0, work = 0, ids = 0, host = 0;
};

ggml_backend_buffer_type_t staging_home(const llama_model & model, ggml_backend_buffer_type_t home) {
    const char * name = std::getenv("LLAMA_MOE_PREFILL_STAGE_DEVICE");
    if (!name || !name[0]) {
        return home;
    }
    auto * dev = ggml_backend_dev_by_name(name);
    if (!dev || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0 ||
        (!model.devices.empty() && std::none_of(model.devices.begin(), model.devices.end(),
                [dev](const llama_device & candidate) { return candidate.dev == dev; }))) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(dev);
}

int32_t staging_token_limit(const llama_model & model) {
    const char * enabled = std::getenv("LLAMA_MOE_PREFILL_STAGE");
    if (!enabled || std::atoi(enabled) == 0 || model.arch != LLM_ARCH_DEEPSEEK41) {
        return LLAMA_MOE_STAGE_DEFAULT_TOKENS;
    }
    const char * limit = std::getenv("LLAMA_MOE_PREFILL_STAGE_MAX_TOKENS");
    if (!limit || !limit[0]) { return LLAMA_MOE_STAGE_DEFAULT_TOKENS; }
    const std::string value(limit);
    if (value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("LLAMA_MOE_PREFILL_STAGE_MAX_TOKENS requires an integer in [33, 8192]");
    }
    unsigned long long tokens;
    try { tokens = std::stoull(value); }
    catch (const std::out_of_range &) { throw std::invalid_argument("LLAMA_MOE_PREFILL_STAGE_MAX_TOKENS exceeds the integer limit"); }
    if (tokens <= LLAMA_MOE_CACHE_MAX_TOKENS || tokens > LLAMA_MOE_STAGE_MAX_TOKENS) {
        throw std::invalid_argument("LLAMA_MOE_PREFILL_STAGE_MAX_TOKENS requires an integer in [33, 8192]");
    }
    return (int32_t) tokens;
}

stage_sizes staging_sizes(const llama_model & model, ggml_backend_buffer_type_t buft,
        const std::vector<const llama_layer *> & layers, int32_t max_tokens) {
    const char * enabled = std::getenv("LLAMA_MOE_PREFILL_STAGE");
    if (!enabled || std::atoi(enabled) == 0 || !buft || model.arch != LLM_ARCH_DEEPSEEK41) {
        return {};
    }
    auto * dev = ggml_backend_buft_get_device(buft);
    if (!dev || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
        return {};
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.host_buffer) {
        return {};
    }
    size_t expert_bytes = 0, dim = 0;
    for (const auto * layer : layers) {
        for (const auto * t : {layer->ffn_up_exps, layer->ffn_gate_exps, layer->ffn_down_exps}) {
            if (host_projection(t) && t->type == GGML_TYPE_Q3_K) {
                expert_bytes = std::max(expert_bytes, t->nb[2]);
                dim = std::max(dim, (size_t) std::max(t->ne[0], t->ne[1]));
            }
        }
    }
    const size_t routes = (size_t) max_tokens * model.hparams.n_expert_used_max();
    if (!expert_bytes || !routes) {
        return {};
    }
    stage_sizes ret;
    ret.host_type = ggml_backend_dev_host_buffer_type(dev);
    ret.weights = 2*(LLAMA_MOE_STAGE_EXPERTS*expert_bytes + 512);
    ret.work = dim*routes;
    ret.ids = 6*routes;
    ret.host = ret.weights + ret.ids*sizeof(int32_t);
    return ret;
}

void staging_tensors(ggml_context * ctx, const stage_sizes & sizes, llama_moe_cache_layer & out) {
    out.stage_weights = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizes.weights);
    out.stage_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sizes.work);
    out.stage_output = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, sizes.work);
    out.stage_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, sizes.ids);
    ggml_set_name(out.stage_weights, "moe_stage_weights");
    ggml_set_name(out.stage_input, "moe_stage_input");
    ggml_set_name(out.stage_output, "moe_stage_output");
    ggml_set_name(out.stage_ids, "moe_stage_ids");
}

void cache_tensors(ggml_context * ctx, const llama_layer & layer, int il, int32_t n_slots,
        llama_moe_cache_layer & out) {
    auto projection = [&](const ggml_tensor * t, const char * name) -> ggml_tensor * {
        if (!host_projection(t)) { return nullptr; }
        auto * cached = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], n_slots + 1);
        ggml_format_name(cached, "%s.%d", name, il);
        return cached;
    };
    out.up_c   = projection(layer.ffn_up_exps, "moe_cache_up");
    out.gate_c = projection(layer.ffn_gate_exps, "moe_cache_gate");
    out.down_c = projection(layer.ffn_down_exps, "moe_cache_down");
    out.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, layer.ffn_up_exps->ne[2]);
    ggml_format_name(out.dev_table, "moe_cache_tbl.%d", il);
}

void moe_obs_cb(const struct ggml_tensor * source, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    if (n_tokens > LLAMA_MOE_CACHE_MAX_TOKENS) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU
    }

    const auto it = mc->by_observe_src.find(source);
    if (it == mc->by_observe_src.end()) {
        return;
    }
    layer_state * ls = &mc->layers[it->second];

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                bool dup = false;
                for (int32_t p : ls->pending) {
                    if (p == id) { dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                }
            }
        }
    }
}

bool upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot, size_t * uploaded_bytes) {
    if (!dst_c) {
        return true;
    }
    const size_t sz = src->nb[2];
    if (!host_projection(src) || !src->data || expert < 0 || expert >= src->ne[2] || slot < 0 || slot >= dst_c->ne[2] - 1 ||
        dst_c->nb[2] != sz || (size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return false;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
    if (uploaded_bytes) {
        *uploaded_bytes += sz;
    }
    return true;
}

void print_profile(const moe_cache & mc) {
    struct totals {
        size_t allocated_bytes = 0;
        uint64_t hits = 0, misses = 0, uploads = 0, published = 0, failed = 0, evicted = 0, bytes = 0;
        int64_t upload_us = 0;
    };
    std::map<ggml_backend_buffer_type_t, totals> devices;
    for (auto * buf : mc.bufs) {
        if (!ggml_backend_buffer_is_host(buf)) {
            devices[ggml_backend_buffer_get_type(buf)].allocated_bytes += ggml_backend_buffer_get_size(buf);
        }
    }
    for (const auto & ls : mc.layers) {
        auto & sum = devices[ggml_backend_buffer_get_type(ls.pub.dev_table->buffer)];
        sum.hits += ls.n_hit;
        sum.misses += ls.n_miss;
        sum.uploads += ls.n_upload;
        sum.published += ls.n_published;
        sum.failed += ls.n_failed;
        sum.evicted += ls.n_evicted;
        sum.bytes += ls.upload_bytes;
        sum.upload_us += ls.upload_us;
    }
    LLAMA_LOG_INFO("moe-cache profile: steps=%" PRIu64 " sync_wait_ms=%.3f cancelled_jobs=%zu\n",
            mc.n_steps, mc.sync_wait_us/1000.0, mc.todo.size());
    for (const auto & [buft, sum] : devices) {
        LLAMA_LOG_INFO("moe-cache profile: device=%s allocated_bytes=%zu hits=%" PRIu64 " misses=%" PRIu64
                " uploads=%" PRIu64 " published=%" PRIu64 " failed=%" PRIu64 " evictions=%" PRIu64
                " upload_bytes=%" PRIu64 " upload_ms=%.3f\n",
                ggml_backend_buft_name(buft), sum.allocated_bytes, sum.hits, sum.misses,
                sum.uploads, sum.published, sum.failed, sum.evicted, sum.bytes, sum.upload_us/1000.0);
    }
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
}

} // namespace

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache_memory_breakdown(const llama_model & model, int32_t n_slots) {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    if (n_slots <= 0 || n_slots == std::numeric_limits<int32_t>::max()) {
        return ret;
    }
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (!model.hparams.no_alloc) {
        if (g_cache && g_cache->model == &model) {
            for (auto * buf : g_cache->bufs) {
                ret[ggml_backend_buffer_get_type(buf)] += ggml_backend_buffer_get_size(buf);
            }
        }
        return ret;
    }

    const auto it = g_metadata.find(&model);
    if (it != g_metadata.end()) {
        return it->second->expected_bytes;
    }
    return ret;
}

static void cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts,
        uint64_t budget_bytes, ggml_backend_dev_t budget_device) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    const bool metadata = model.hparams.no_alloc;
    if (metadata ? g_metadata.count(&model) != 0 : g_init_done) {
        const auto * previous = metadata ? g_metadata.at(&model) : g_cache;
        if (previous && previous->model == &model && (budget_bytes || previous->budget_bytes) &&
            (previous->budget_bytes != budget_bytes || previous->budget_device != budget_device)) {
            throw std::invalid_argument("MoE cache: cannot change a live owner budget");
        }
        return;
    }
    [&]() {
        if (n_slots <= 0 && budget_bytes == 0) {
            return;
        }
        if (n_slots == std::numeric_limits<int32_t>::max()) {
            LLAMA_LOG_WARN("%s: slot count leaves no room for the dummy slot - cache disabled\n", __func__);
            return;
        }

        // Prefer the router's device; a resident projection is also a cache home.
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!cacheable_layer(l) || (!metadata && (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data))) {
                continue;
            }
            const auto home = cache_home(l);
            if (!budget_bytes || ggml_backend_buft_get_device(home) == budget_device) {
                groups[home].push_back({(int) il, &l});
            }
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            return;
        }

        const int32_t stage_max_tokens = staging_token_limit(model);
        if (budget_bytes) {
            // Size the same typed tensors used below; descriptor-only probes never allocate GPU payloads.
            auto footprint = [&](int32_t slots) {
                size_t total = 0;
                for (const auto & group : groups) {
                    auto * ctx = ggml_init({ggml_tensor_overhead()*(group.second.size()*4 + 8), nullptr, true});
                    if (!ctx) { throw std::runtime_error("MoE cache: failed to size the byte budget"); }
                    std::vector<const llama_layer *> layers;
                    for (const auto & c : group.second) {
                        llama_moe_cache_layer tensors;
                        cache_tensors(ctx, *c.l, c.il, slots, tensors);
                        layers.push_back(c.l);
                    }
                    const auto stage_buft = staging_home(model, group.first);
                    const auto sizes = staging_sizes(model, stage_buft, layers, stage_max_tokens);
                    if (stage_buft == group.first && sizes.host_type) {
                        llama_moe_cache_layer workspace;
                        staging_tensors(ctx, sizes, workspace);
                    }
                    const size_t size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, group.first);
                    ggml_free(ctx);
                    if (size > std::numeric_limits<size_t>::max() - total) {
                        throw std::overflow_error("MoE cache: byte budget size overflow");
                    }
                    total += size;
                }
                return total;
            };
            int32_t high = std::numeric_limits<int32_t>::max() - 1;
            for (const auto & group : groups) {
                for (const auto & c : group.second) {
                    high = std::min<int64_t>(high, c.l->ffn_up_exps->ne[2]);
                }
            }
            int32_t low = 0;
            while (low < high) {
                const int32_t mid = low + (high - low + 1)/2;
                if (footprint(mid) <= budget_bytes) { low = mid; }
                else { high = mid - 1; }
            }
            if (low == 0) {
                throw std::invalid_argument("MoE cache: byte budget cannot hold one slot per layer plus overhead");
            }
            n_slots = low;
            LLAMA_LOG_INFO("moe-cache: %s budget %.2f MiB, planned %.2f MiB, %d fixed slots/layer%s\n",
                    ggml_backend_dev_name(budget_device), budget_bytes/1048576.0, footprint(n_slots)/1048576.0,
                    n_slots, metadata ? " (metadata)" : "");
        }

        auto * mc = new moe_cache();
        mc->model = &model;
        mc->n_slots = n_slots;
        const char * profile = std::getenv("LLAMA_MOE_CACHE_PROFILE");
        mc->profile = profile && std::atoi(profile) != 0;
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }

        mc->budget_bytes = budget_bytes;
        mc->budget_device = budget_device;

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_tensors = [&](ggml_context * ctx, ggml_backend_buffer_type_t buft) {
            if (!metadata) {
                return ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            }
            auto * buf = ggml_backend_buft_alloc_buffer(buft, 0);
            if (buf) {
                mc->expected_bytes[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
                for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
                    t->buffer = buf;
                }
            }
            return buf;
        };

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*4 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->pub.n_slots  = n_slots;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                    ls->pub.observe_src = host_projection(c.l->ffn_gate_exps) ? c.l->ffn_gate_exps :
                            host_projection(c.l->ffn_up_exps) ? c.l->ffn_up_exps : c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
                    if (dev) {
                        auto get_max_batch = (ggml_backend_dev_mul_mat_id_repeated_ids_max_batch_t) ggml_backend_reg_get_proc_address(
                                ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_mul_mat_id_repeated_ids_max_batch");
                        if (get_max_batch) {
                            ls->pub.n_batch = LLAMA_MOE_CACHE_MAX_TOKENS;
                            for (const auto * t : {u, g, d}) {
                                if (host_projection(t)) {
                                    ls->pub.n_batch = std::min(ls->pub.n_batch, std::max(1, get_max_batch(dev, t->type)));
                                }
                            }
                        }
                    }
                    cache_tensors(ctx, *c.l, c.il, n_slots, ls->pub);
                }
            }

            if (!tables_only) {
                std::vector<const llama_layer *> layers;
                for (const auto & c : cands) { layers.push_back(c.l); }
                const auto stage_buft = staging_home(model, buft);
                const auto sizes = staging_sizes(model, stage_buft, layers, stage_max_tokens);
                if (sizes.host_type) {
                    auto * pinned = ggml_init({ggml_tensor_overhead()*2, nullptr, true});
                    if (!pinned) { return false; }
                    mc->ctxs.push_back(pinned);
                    auto * host_pool = ggml_new_tensor_1d(pinned, GGML_TYPE_I8, sizes.host);
                    ggml_set_name(host_pool, "moe_stage_host");
                    auto * host_buf = alloc_tensors(pinned, sizes.host_type);
                    if (!host_buf) { return false; }
                    mc->bufs.push_back(host_buf);
                    if (ggml_backend_buffer_get_type(host_buf) != sizes.host_type) { return false; }
                    if (!metadata) { ggml_backend_buffer_clear(host_buf, 0); }
                    llama_moe_cache_layer workspace;
                    auto * workspace_ctx = ctx;
                    if (stage_buft != buft) {
                        workspace_ctx = ggml_init({ggml_tensor_overhead()*8, nullptr, true});
                        if (!workspace_ctx) { return false; }
                        mc->ctxs.push_back(workspace_ctx);
                    }
                    staging_tensors(workspace_ctx, sizes, workspace);
                    if (workspace_ctx != ctx) {
                        auto * stage_buf = alloc_tensors(workspace_ctx, stage_buft);
                        if (!stage_buf) { return false; }
                        mc->bufs.push_back(stage_buf);
                        ggml_backend_buffer_set_usage(stage_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                        if (!metadata) { ggml_backend_buffer_clear(stage_buf, 0); }
                    }
                    for (auto & ls : mc->layers) {
                        if (std::any_of(cands.begin(), cands.end(), [&ls](const cand & c) { return c.il == ls.pub.il; })) {
                            ls.pub.stage_weights = workspace.stage_weights;
                            ls.pub.stage_input = workspace.stage_input;
                            ls.pub.stage_output = workspace.stage_output;
                            ls.pub.stage_ids = workspace.stage_ids;
                            ls.pub.stage_host = host_pool;
                            ls.pub.stage_max_tokens = stage_max_tokens;
                        }
                    }
                    LLAMA_LOG_INFO("moe-cache: prefill staging on %s, max_tokens=%d, device scratch %.1f MiB, pinned scratch %.1f MiB\n",
                            ggml_backend_buft_name(stage_buft), stage_max_tokens, (sizes.weights + 2*sizes.work*sizeof(float) + sizes.ids*sizeof(int32_t))/1048576.0,
                            sizes.host/1048576.0);
                }
            }

            ggml_backend_buffer_t buf = alloc_tensors(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            if (!metadata) { ggml_backend_buffer_clear(buf, 0); }
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            mc->bufs.push_back(buf);
            return true;
        };

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            ok = alloc_group(g.first, g.second, /*tables_only=*/false);
        }

        bool over_budget = false;
        if (ok && budget_bytes) {
            size_t actual = 0;
            if (metadata) {
                for (const auto & b : mc->expected_bytes) {
                    if (!ggml_backend_buft_is_host(b.first) && ggml_backend_buft_get_device(b.first) == budget_device) {
                        actual += b.second;
                    }
                }
            } else {
                for (auto * b : mc->bufs) {
                    const auto buft = ggml_backend_buffer_get_type(b);
                    if (!ggml_backend_buft_is_host(buft) && ggml_backend_buft_get_device(buft) == budget_device) {
                        actual += ggml_backend_buffer_get_size(b);
                    }
                }
            }
            over_budget = actual > budget_bytes;
        }
        if (!ok || over_budget) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            if (over_budget) { throw std::runtime_error("MoE cache: allocation exceeded its byte budget"); }
            return;
        }

        if (metadata) {
            for (size_t i = 0; i < mc->layers.size(); ++i) {
                mc->by_up_src[mc->layers[i].pub.up_src] = i;
            }
            g_metadata.emplace(&model, mc);
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(n_slots, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(n_slots, 0);
            ls.slot_in_flight.assign(n_slots, false);
            ls.expert_in_flight.assign(n_expert, false);

            std::vector<int32_t> dummy(n_expert, n_slots);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            mc->by_observe_src[ls.pub.observe_src] = &ls - mc->layers.data();
            for (const auto * t : {ls.pub.up_c, ls.pub.gate_c, ls.pub.down_c}) {
                if (t) {
                    vram += ggml_nbytes(t);
                }
            }
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert, GPU batch limit %d\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2], ls.pub.n_batch);
        }

        mc->worker = std::thread([mc]() {
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || !mc->todo.empty(); });
                    if (mc->stop) {
                        return;
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                }
                auto & ls = mc->layers[j.layer_idx];
                const int64_t start = mc->profile ? ggml_time_us() : 0;
                size_t bytes = 0;
                size_t * uploaded_bytes = mc->profile ? &bytes : nullptr;
                j.success = upload_slice(ls.pub.up_c, ls.pub.up_src, j.expert, j.slot, uploaded_bytes) &&
                        upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot, uploaded_bytes) &&
                        upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot, uploaded_bytes);
                {
                    std::lock_guard<std::mutex> lk(mc->wmtx);
                    if (mc->profile) {
                        ls.n_upload += j.success;
                        ls.n_failed += !j.success;
                        ls.upload_bytes += bytes;
                        ls.upload_us += ggml_time_us() - start;
                    }
                    mc->done.push_back(j);
                }
            }
        });

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(), n_slots, mc->max_inserts, vram/1024.0/1024.0);
    }();
}

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts) {
    cache_init(model, n_slots, max_inserts, 0, nullptr);
}

int32_t llama_moe_cache_init_budget(const llama_model & model, uint64_t bytes,
        ggml_backend_dev_t device, int32_t max_inserts) {
    if (!bytes || bytes > std::numeric_limits<size_t>::max() || !device ||
        ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
        std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0) {
        throw std::invalid_argument("MoE cache: a positive byte budget and an available CUDA device are required");
    }
    cache_init(model, 0, max_inserts, bytes, device);
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    const auto metadata = g_metadata.find(&model);
    const auto * mc = metadata != g_metadata.end() ? metadata->second : g_cache;
    return mc && mc->model == &model ? mc->n_slots : 0;
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    for (const auto & entry : g_metadata) {
        const auto it = entry.second->by_up_src.find(up_exps);
        if (it != entry.second->by_up_src.end()) {
            return &entry.second->layers[it->second].pub;
        }
    }
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step(const llama_model & model, ggml_backend_sched_t sched) {
    moe_cache * mc = g_cache;
    if (!mc || mc->model != &model) {
        return;
    }

    const int64_t start = mc->profile ? ggml_time_us() : 0;
    ggml_backend_sched_synchronize(sched);

    // 1) publish completed uploads (sync point: no graph is executing)
    {
        std::lock_guard<std::mutex> wlk(mc->wmtx);
        std::lock_guard<std::mutex> lk(mc->mtx);
        if (mc->profile) {
            mc->sync_wait_us += ggml_time_us() - start;
        }
        for (const auto & j : mc->done) {
            auto & ls = mc->layers[j.layer_idx];
            ls.slot_in_flight[j.slot] = false;
            ls.expert_in_flight[j.expert] = false;
            if (!j.success) {
                continue;
            }
            ls.slot_expert[j.slot]     = j.expert;
            ls.expert_slot[j.expert]   = j.slot;
            ls.slot_last_use[j.slot]   = ++mc->clock;
            set_table_entry(ls.pub, j.expert, j.slot);
            if (mc->profile) {
                ++ls.n_published;
            }
        }
        mc->done.clear();
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.pending.empty()) {
            continue;
        }

        int budget = mc->max_inserts;
        for (auto it = ls.pending.rbegin(); it != ls.pending.rend() && budget > 0; ++it) {
            const int32_t id = *it;
            if (ls.expert_slot[id] >= 0 || ls.expert_in_flight[id]) {
                continue;
            }

            // victim: an empty non-in-flight slot if any, else the LRU non-in-flight slot
            int32_t slot = -1;
            uint64_t best = UINT64_MAX;
            for (int32_t s = 0; s < ls.pub.n_slots; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) { slot = s; break; }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; slot = s; }
            }
            if (slot < 0) {
                break; // every slot is in flight; try again next step
            }

            const int32_t victim = ls.slot_expert[slot];
            if (victim >= 0) {
                ls.expert_slot[victim] = -1;
                ls.slot_expert[slot]   = -1;
                set_table_entry(ls.pub, victim, ls.pub.n_slots);
                if (mc->profile) {
                    ++ls.n_evicted;
                }
            }
            ls.slot_in_flight[slot] = true;
            ls.expert_in_flight[id] = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            mc->todo.push_back({li, id, slot});
            --budget;
        }
        ls.pending.clear();
    }
    mc->wcv.notify_one();

    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        LLAMA_LOG_DEBUG("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
    }
}

void llama_moe_cache_free(const llama_model & model) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    const auto metadata = g_metadata.find(&model);
    if (metadata != g_metadata.end()) {
        auto * mc = metadata->second;
        for (auto * buf : mc->bufs) { ggml_backend_buffer_free(buf); }
        for (auto * ctx : mc->ctxs) { ggml_free(ctx); }
        delete mc;
        g_metadata.erase(metadata);
        return;
    }
    moe_cache * mc = g_cache;
    if (!mc || mc->model != &model) {
        return;
    }

    ggml_set_moe_obs_callback(nullptr, nullptr);
    g_cache = nullptr;
    {
        std::lock_guard<std::mutex> lock(mc->wmtx);
        mc->stop = true;
    }
    mc->wcv.notify_one();
    mc->worker.join();

    if (mc->profile) {
        print_profile(*mc);
    }

    for (auto * buf : mc->bufs) {
        ggml_backend_buffer_free(buf);
    }
    for (auto * ctx : mc->ctxs) {
        ggml_free(ctx);
    }
    delete mc;
    g_init_done = false;
}
