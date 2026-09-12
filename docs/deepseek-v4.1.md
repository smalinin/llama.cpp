# DeepSeek-V4.1-Flash

This document describes the text-only DeepSeek-V4.1-Flash support in this branch.
It covers conversion, loading, sparse attention, Engram, the V4.1
hyper-connection schedule, and the official text conversation protocol.

## Current scope

The following paths have been validated:

- x86-64 CPU inference;
- CUDA inference and layer split on NVIDIA `sm_86` and `sm_89` GPUs;
- raw completion and the OpenAI-compatible text chat API;
- the two-level candidate block mask on CPU and CUDA;
- full-Q2 Wikitext prefill at 16640, 32768, and 65536 tokens;
- a 32768-context server lifecycle and completion.

Other backends may use the generic graph operations, but have not been
validated for this model. A 128K full-model prefill was not run because 64K
took 28 minutes after memory fitting moved more work to the CPU. Context shift
is not supported.

The vision tower, multimodal projector, MTP head, and DSpark draft model are not
mapped. Use this implementation as a text-only target model.

## Provenance

The initial implementation was integrated from the following pinned sources
and then adapted and tested against this branch:

- [converter PR](https://github.com/ggml-org/llama.cpp/pull/28696):
  `vcruz305/llama.cpp` commit
  `b12818a24407175d941e9299e7b5fb7874a654d9`;
- [runtime prototype](https://github.com/vcruz305/llama.cpp/commit/f37da57110ebbe07e982a934f2d444d9fd30eb09):
  `vcruz305/llama.cpp` commit
  `f37da57110ebbe07e982a934f2d444d9fd30eb09`;
- [official model and inference reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277):
  `deepseek-ai/DeepSeek-V4.1-Flash` revision
  `dba1be0a40aa45a94ad051997016db3960a90277`.

The official checkpoint inference code is the primary numerical and prompt
format reference. The following sources were used as independent cross-checks:

- [vLLM](https://github.com/vllm-project/vllm/commit/912dfb37581b98c0b6eaeca5758d1e1462b7feac)
  commit `912dfb37581b98c0b6eaeca5758d1e1462b7feac`;
- [SGLang](https://github.com/sgl-project/sglang/commit/0d5e663b8f8d80a6caec2a7f7ce4eed6394756b7)
  commit `0d5e663b8f8d80a6caec2a7f7ce4eed6394756b7`;
- [JigSawPT runtime](https://github.com/JigSawPT/llama.cpp/commit/2a171529a2050b574870ed4ae1a8cc1ec683790a)
  commit `2a171529a2050b574870ed4ae1a8cc1ec683790a`.

## Build

CPU Release build:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build build --config Release -j --target \
    llama-completion llama-server llama-quantize
```

CUDA Release build for the GPU architectures used during validation:

```sh
cmake -B build-cuda \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA=ON \
    -DCMAKE_CUDA_ARCHITECTURES='86-real;89-real'
cmake --build build-cuda --config Release -j --target \
    llama-completion llama-server llama-quantize
```

Choose `CMAKE_CUDA_ARCHITECTURES` for the deployment GPUs. Architectures other
than 86 and 89 were not part of the model validation described below.

## Convert and quantize

Convert a local copy of the official checkpoint with the converter from this
branch. Conversion needs the complete text checkpoint, config, tokenizer, and
safetensors index:

```sh
python3 convert_hf_to_gguf.py /path/to/DeepSeek-V4.1-Flash \
    --outfile /path/to/DeepSeek-V4.1-Flash-BF16.gguf \
    --outtype bf16 \
    --split-max-size 40G
```

The converter exports text weights only. It writes all nine
`deepseek41.engram.*` metadata entries with explicit integer array types,
quantizes the very large Engram embedding tables to Q8_0 in bounded chunks,
embeds the V4.1 chat template, and writes the candidate source layer, block
size, and top-k block count as one required group when they are present in the
checkpoint config. Conversion must fail if required Engram fields or a partial
candidate configuration cannot be written.

Quantize from a corrected high-precision GGUF. The quantizer keeps
`engram_q`, `engram_k`, `hc_attn_fn`, and `hc_ffn_fn` unquantized because these
small tensors participate in numerically sensitive scale and mixing paths:

```sh
./build/bin/llama-quantize --keep-split \
    /path/to/DeepSeek-V4.1-Flash-BF16-00001-of-N.gguf \
    /path/to/DeepSeek-V4.1-Flash-Q2_K Q2_K
```

When `--split-max-size` creates multiple conversion shards, pass the first
shard to the quantizer and use `--keep-split`. The output argument above is a
base name without `.gguf`; the quantizer adds the numbered split suffixes.

Use Q2_K or a higher-precision quantization for runtime work. Q1_0 execution is
not a useful quality target for this model because the quantized token
embedding loses almost all magnitude information.

### Pre-fix community GGUF files

GGUF files produced before converter commit
`b12818a24407175d941e9299e7b5fb7874a654d9` may have Engram metadata under the
wrong architecture prefix and may omit five arrays after a hidden integer
overflow. Reconvert them when possible.

Early community GGUF files also omit the three candidate-mask keys. The loader
uses the checkpoint-specific official values `source=20`, `block_size=8`, and
`top_k_blocks=2048` only when the file has the known 40-layer, 1M-context,
top-k-512 configuration. Other files without the complete metadata group keep
the conservative 16384-token runtime limit.

A metadata-only repair can be useful for loader and graph bring-up. It must
rename the old keys to the `deepseek41.engram.*` namespace and add
`multipliers`, `primes`, `offsets`, `token_map`, and `pad_id` with the same
types and values generated from the pinned official tokenizer. This does not
repair quantized weights and should not be published as a quality conversion.

For local diagnostics, the four `engram_q` and `engram_k` tensors at layers 1
and 14 can be restored as BF16 from the official checkpoint shards selected by
`model.safetensors.index.json`. Tensor offsets, 32-byte GGUF alignment, split
metadata, source hashes, shape `[5120, 4]`, and the rewritten payload hashes
must all be verified. This overlay improves the observed Q2 answer, but old
quantized mHC projections remain, so a fresh conversion is still the required
production path.

## Run

The two Engram embedding tables are about 30.76 GiB each in the tested Q2 set.
Keep them mmap-backed and lazy-read instead of making them resident:

```sh
./build-cuda/bin/llama-completion \
    -m /path/to/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
    -c 8192 -b 2048 -ub 256 \
    -ngl auto -sm layer --fit on -fitc 8192 -fitt 2048 \
    -lm mmap -lzm auto --no-context-shift \
    --no-conversation --no-display-prompt --simple-io \
    -p 'The chemical symbol for gold is' -n 32 --temp 0 -s 1234
```

`--lazy-mode auto` lazy-reads eligible tensors larger than 4 GiB and requires
mmap. Use `--lazy-mode on` to force the same policy for eligible smaller test
tables. Do not combine this model with `--load-mode none`, `--no-mmap`, or
`--lazy-mode off` unless enough resident RAM is available and the memory impact
is intentional.

For prompt batches of at least 32 tokens, the runtime computes all Engram row
IDs first and gives the operating system one page-aligned, de-duplicated set of
read-ahead hints before gathering the mmap-backed rows. This reduces blocking
major page faults on a cold page cache without making the complete tables
resident. It is a best-effort hint and does not change the rows or tensor
arithmetic. Small decode batches skip it by default. For diagnostics, set
`LLAMA_DSV41_ENGRAM_PREFETCH=off` to disable the hints or `=on` to issue them
for every batch size; leave the variable unset for the default automatic mode.

For the text chat API:

```sh
./build-cuda/bin/llama-server \
    -m /path/to/DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf \
    -c 8192 -b 2048 -ub 256 \
    -ngl auto -sm layer --fit on -fitc 8192 -fitt 2048 \
    -lm mmap -lzm auto --no-context-shift \
    --jinja --reasoning-format deepseek \
    --no-reasoning-preserve --no-prefill-assistant \
    --chat-template-kwargs '{"reasoning_effort":80,"enable_thinking":true}'
```

Newly converted files embed the V4.1 template. Older community GGUF files may
embed the V4 template instead; for those files, pass
`--chat-template-file models/templates/deepseek-ai-DeepSeek-V4.1.jinja`.
Numeric `reasoning_effort=80` and thinking on/off were checked against the
official encoder. Named effort aliases differ between external runtimes, so
use a numeric value when exact cross-runtime behavior matters.

## Validation summary

The MVP was validated with the following fixed suites:

| Suite | Result |
| --- | ---: |
| Engram and hyper-connection numerical checks | 43/43 |
| Compressed KV and sparse attention checks | 154/154 |
| Tiny end-to-end CPU/CUDA checks | 30/30 |
| Full repaired Q2 operational checks | 27/27 |
| Cross-architecture regression matrix | 46/46 |
| Official encoding and server chat checks | 60/60 |
| Candidate mask, graph reuse, and context lifecycle | 91/91 |
| Full-Q2 long prefill, state/rollback, and 32K server | 56/56 |

The full Q2 test used seven shards, 246.34 GiB, 748.49 billion parameters, and
1046 tensors. All 41 layers were offloaded across six NVIDIA GPUs. The process
used 197092 MiB of VRAM versus a 194819 MiB fit estimate, with a minimum device
margin of 4614 MiB. Lazy Engram mappings totaled 60.08 GiB while the idle server
RSS was 5.65 GiB. The measured peak RSS during loading/offload was 185.51 GiB.

On that machine, 32-token raw completion produced 38.72-39.89 prompt tokens/s
and 33.95-40.03 generated tokens/s after roughly 54 seconds of warm-cache model
loading. These are smoke measurements on a heterogeneous six-GPU system, not a
portable benchmark.

With the final Q2 quality overlay, a server configured for 32768 context
generated 32 tokens at 32.73 tokens/s. Single Wikitext-2 perplexity passes took
117.26 seconds at 16640 tokens, 233.70 seconds at 32768, and 1700.64 seconds at
65536. The sharp 64K slowdown occurred after automatic memory fitting placed
more work on the CPU. These different-length corpus prefixes are operational
and finite-likelihood checks; their PPL values are not directly comparable as
a quality trend.

The repaired community Q2 with quantized `engram_q/k` loaded and generated but
gave a weak short answer. Replacing those four tensors with official BF16 data
produced the expected `Au` token within 32 generated tokens. Corpus NLL and a
full quality benchmark against the official reference have not been completed.

## Known differences from the official reference

- Candidate selection is implemented and checked exactly against the official
  algorithm, including a partial final block. Full-Q2 prefill is validated
  through 64K, but 128K and reference-runtime NLL parity remain unmeasured.
- The 32K multi-GPU runs grew the CUDA2 compute buffer by 16 MiB beyond the
  startup estimate. This stayed inside the 2048 MiB fit target, did not affect
  deterministic PPL, and did not recur at 16.6K or 64K; memory-fit accounting
  for intermediate graph shapes remains an optimization item.
- Vision, multimodal input, MTP, and DSpark are not exported or executed.
- Candidate masks are recomputed from the current compressed scores, so normal
  cache rollback needs no separate persistent mask state. Context shift remains
  disabled and has not been validated with candidate selection.
- The model-side text tool-call syntax is rendered and parsed, but llama-server
  does not execute external tools for the caller.
- The full community Q2 smoke is evidence for loader, graph, memory, and
  lifecycle behavior. It is not evidence of reference-level model quality.
