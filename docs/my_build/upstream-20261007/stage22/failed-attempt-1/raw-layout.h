#pragma once
#include "llama-batch.h"
#include "llama-kv-cache-dsv4.h"
#include "llama-kv-cache-iswa.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ds22 {
struct RawPlan {
    std::vector<int> extents;
    std::vector<uint32_t> indices;
    std::vector<llama_pos> positions;
    int before_max = 0;
};
static RawPlan raw_plan(llama_memory_t memory, const llama_batch & batch) {
    auto * dsv4 = dynamic_cast<llama_kv_cache_dsv4 *>(memory);
    if (!dsv4 || batch.n_tokens < 1 || batch.n_tokens > 4 || !batch.pos || !batch.token) throw std::runtime_error("unsupported raw plan batch");
    auto * raw = dsv4->get_raw()->get_swa();
    const auto & cells = raw->get_cells(0);
    RawPlan plan;
    plan.before_max = cells.used_max_p1();
    std::vector<llama_pos> positions(cells.size(), -1);
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i)) continue;
        if (cells.seq_count(i) != 1 || cells.seq_get(i) != 0) throw std::runtime_error("raw plan requires one sequence");
        positions[i] = cells.pos_get(i);
    }
    llama_batch_allocr allocator(1);
    auto ubatch = allocator.ubatch_reserve(batch.n_tokens, 1);
    ubatch.seq_id_unq[0] = 0;
    for (int i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != 0 || batch.pos[i] != batch.pos[0]+i) throw std::runtime_error("raw plan requires contiguous positions");
        ubatch.token[i] = batch.token[i];
        ubatch.pos[i] = batch.pos[i];
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i][0] = 0;
    }
    const auto slot = raw->find_slot(ubatch, false);
    if (slot.empty() || slot.n_stream() != 1 || slot.idxs[0].size() != size_t(batch.n_tokens)) throw std::runtime_error("raw plan slot failed");
    plan.indices = slot.idxs[0];
    llama_pos purge = -1;
    for (int col = 0; col < batch.n_tokens; ++col) {
        const auto index = plan.indices[col];
        purge = std::max(purge, positions[index]);
        positions[index] = batch.pos[col];
        int used_max = 0;
        for (uint32_t i = 0; i < positions.size(); ++i) {
            if (positions[i] >= 0 && positions[i] <= purge) positions[i] = -1;
            if (positions[i] >= 0) used_max = i+1;
        }
        plan.extents.push_back(std::min<int>(cells.size(), std::max(256, (used_max+255)/256*256)));
        plan.positions.push_back(batch.pos[col]);
    }
    return plan;
}
static void check_raw_plan(llama_memory_t memory, const RawPlan & plan, int source_extent) {
    auto * dsv4 = dynamic_cast<llama_kv_cache_dsv4 *>(memory);
    if (!dsv4 || plan.extents.empty() || plan.extents.back() != source_extent) throw std::runtime_error("raw plan source extent mismatch");
    const auto & cells = dsv4->get_raw()->get_swa()->get_cells(0);
    for (size_t col = 0; col < plan.indices.size(); ++col) {
        const auto index = plan.indices[col];
        if (cells.is_empty(index) || cells.pos_get(index) != plan.positions[col]) throw std::runtime_error("raw plan physical index mismatch");
    }
}
}
