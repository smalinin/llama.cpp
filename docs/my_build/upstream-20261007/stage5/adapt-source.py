#!/usr/bin/env python3
"""Adapt the upstream indexer patches to the local QSA graph."""
from pathlib import Path

repo = Path('/home/sergei/Github/llama.cpp')
p = repo / 'src/models/qwen4exp.cpp'
s = p.read_text()
s = s.replace('// bad metadata must be catchable:', '''static bool qwen4exp_fused_lid_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("QWEN4EXP_FUSED_LID");
        return env != nullptr && std::atoi(env) != 0;
    }();

    return enabled;
}

// bad metadata must be catchable:''', 1)
start = s.index('        // Rectify each head dot product')
end = s.index('        cb(score, "indexer_score_blk", il);', start)
legacy = s[start:end]
legacy = legacy.replace('        ggml_tensor * score =', '        score =', 1)
legacy = '\n'.join('    ' + line if line else line for line in legacy.splitlines())
fused = '''        ggml_tensor * score = nullptr;
        if (fused_lid) {
            ggml_tensor * weights = ggml_fill(ctx0,
                    ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_idx_h, n_tps, 1, n_stream), 1.0f);
            ggml_tensor * mask = ggml_cast(ctx0, ggml_scale(ctx0, inp->blk_bias, 0.0f), GGML_TYPE_F16);
            mask = ggml_reshape_4d(ctx0, mask, n_blocks, n_tps, 1, n_stream);
            score = ggml_lightning_indexer(ctx0,
                    ggml_reshape_4d(ctx0, q, idx_dim, n_idx_h, n_tps, n_stream),
                    ggml_reshape_4d(ctx0, pooled, idx_dim, 1, n_blocks, n_stream), weights, mask);
            res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il});
            // Keep the local unscaled scores and finite F32 visibility bias.
            score = ggml_add(ctx0, ggml_reshape_3d(ctx0, score, n_blocks, n_tps, n_stream), inp->blk_bias);
        } else {
'''
s = s[:start] + fused + legacy + '\n        }\n' + s[end:]
s = s.replace('''        cb(pooled, "indexer_k", il);

        ggml_tensor * q = build_lora_mm''', '''        cb(pooled, "indexer_k", il);

        const bool fused_lid = cparams.fused_lid && qwen4exp_fused_lid_enabled() && idx_dim == 128 && n_idx_h == 4;
        if (fused_lid) {
            // Keep key pooling before the query projection to release its temporary buffers.
            ggml_build_forward_expand(gf, pooled);
        }

        ggml_tensor * q = build_lora_mm''', 1)
p.write_text(s)

p = repo / 'ggml/src/ggml-cuda/lightning-indexer.cu'
s = p.read_text().replace('#include <cstdlib>', '#include <cstdlib>\n#include <type_traits>')
start = s.index('// one block scores a tile of K_VECS_PER_BLOCK')
end = s.index('#define LIGHTNING_INDEXER_CASE', start)
tile = s[start:end]
tile = tile.replace('// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD',
                    '// staged at their storage precision and the queries of every head in float, each thread owns KEYS_PER_THREAD')
tile = tile.replace('    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];',
                    '    using key_pair = std::conditional_t<TYPE_K == GGML_TYPE_F32 || TYPE_K == GGML_TYPE_BF16, float2, half2>;\n    __shared__ key_pair k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];')
tile = tile.replace('        half2 lo = make_half2(0.0f, 0.0f);\n        half2 hi = lo;',
                    '        key_pair lo{}, hi{};')
tile = tile.replace('                lo = make_half2(v.x, v.y);\n                hi = make_half2(v.z, v.w);',
                    '''                if constexpr (TYPE_K == GGML_TYPE_F32 || TYPE_K == GGML_TYPE_BF16) {
                    lo = make_float2(v.x, v.y);
                    hi = make_float2(v.z, v.w);
                } else {
                    lo = make_half2(v.x, v.y);
                    hi = make_half2(v.z, v.w);
                }''')
tile = tile.replace('            k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);',
                    '''            if constexpr (TYPE_K == GGML_TYPE_F32 || TYPE_K == GGML_TYPE_BF16) {
                k_val[j] = k_shared[kl + j*KEY_LANES][c];
            } else {
                k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
            }''')
s = s[:start] + tile + s[end:]
# Float keys use a smaller tile to stay below 48 KiB shared memory.
start = s.index('    } else if (n_embd == 128 && n_head == 4 && n_batch >=')
end = s.index('    } else if (n_embd == 128 && n_head == 4)', start)
branch = s[start:end]
branch = branch.replace('        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)\n', '')
branch = branch.replace('        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)\n', '')
float_branch = branch[:branch.index('        LIGHTNING_INDEXER_CASE')]
float_branch = float_branch.replace('n_batch >= LIGHTNING_INDEXER_TILE_TOKENS)',
                                   'n_batch >= LIGHTNING_INDEXER_TILE_TOKENS &&\n            (k->type == GGML_TYPE_F32 || k->type == GGML_TYPE_BF16))')
float_branch = float_branch.replace('K_VECS_PER_BLOCK = 64', 'K_VECS_PER_BLOCK = 32')
float_branch += '''        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        GGML_ABORT("fatal error");
'''
s = s[:start] + float_branch + branch + s[end:]
p.write_text(s)
print('Adapted local QSA and retained F32/BF16 key precision.')
