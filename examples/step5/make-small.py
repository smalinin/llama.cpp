import argparse
from pathlib import Path
import numpy as np
import gguf


parser = argparse.ArgumentParser(description='Create a small deterministic Step-5 cache test model')
parser.add_argument('--output', type=Path, default=Path('/tmp/step5-small-csa.gguf'))
path = parser.parse_args().output
w = gguf.GGUFWriter(path, 'step35')
for key, value in {'context_length': 8192, 'embedding_length': 64, 'block_count': 4,
                   'feed_forward_length': 128, 'expert_feed_forward_length': 128,
                   'attention.head_count': 64, 'attention.head_count_kv': 4,
                   'attention.key_length': 192, 'attention.value_length': 192,
                   'rope.dimension_count': 64, 'rope.dimension_count_swa': 192,
                   'attention.sliding_window': 512, 'vocab_size': 128}.items():
    w.add_uint32('step35.' + key, value)
w.add_float32('step35.attention.layer_norm_rms_epsilon', 1e-5)
w.add_float32('step35.rope.freq_base', 1e7)
w.add_float32('step35.rope.freq_base_swa', 1e4)
w.add_sliding_window_pattern([True, True, True, False])
w.add_tokenizer_model('no_vocab')
rng = np.random.default_rng(1)


def tensor(name, shape, constant=None):
    data = rng.normal(0, 0.02, shape).astype(np.float32) if constant is None else np.full(shape, constant, dtype=np.float32)
    w.add_tensor(name, data)


tensor('token_embd.weight', (128, 64))
tensor('output.weight', (128, 64))
tensor('output_norm.weight', (64,), 1)
for layer in range(4):
    prefix = f'blk.{layer}.'
    tensor(prefix + 'attn_norm.weight', (64,), 1)
    tensor(prefix + 'attn_q.weight', (64 * 192, 64))
    tensor(prefix + 'attn_k.weight', (4 * 192, 64))
    tensor(prefix + 'attn_v.weight', (4 * 192, 64))
    tensor(prefix + 'attn_output.weight', (64, 64 * 192))
    tensor(prefix + 'attn_q_norm.weight', (192,), 1)
    tensor(prefix + 'attn_k_norm.weight', (192,), 1)
    tensor(prefix + 'ffn_norm.weight', (64,), 1)
    tensor(prefix + 'ffn_gate.weight', (128, 64))
    tensor(prefix + 'ffn_up.weight', (128, 64))
    tensor(prefix + 'ffn_down.weight', (64, 128))
tensor('blk.3.indexer.q_proj.weight', (256 * 16, 64))
tensor('blk.3.indexer.k_proj.weight', (256, 64))
tensor('blk.3.indexer.z_proj.weight', (256, 64))
tensor('blk.3.indexer.proj.weight', (16, 64))
tensor('blk.3.indexer.q_norm.weight', (256,), 0)
tensor('blk.3.indexer.k_norm.weight', (256,), 1)
tensor('blk.3.indexer.k_norm.bias', (256,), 0)
tensor('blk.3.attn_ssmax_s', (64,), 0.08496)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(path)
