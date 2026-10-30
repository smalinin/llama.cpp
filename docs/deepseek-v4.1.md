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

The list above records the initial text-only validation, not the complete scope
of the current branch. Vision/projector conversion and DSpark support were added
later. DSpark uses a separate sidecar GGUF loaded through the `dflash`
architecture; BF16 and MXFP4 sidecars are supported. Target HC means are extracted
at the sidecar's configured layers, encoded into its KV cache, and used by its
Markov and confidence heads. See `src/models/dflash.cpp` and
`common/speculative.cpp` for the current implementation. This does not imply
native `draft-mtp` support on the `deepseek41` target architecture.

## Experimental DSpark admission

Two environment variables are available for `draft-dspark` experiments:

- `LLAMA_DSPARK_CUM_P_MIN`: a finite value in `[0, 1]`, default `0` (off).
  Stop admission when the product of confidence scores falls below this value.
  The individual `--spec-draft-p-min` floor still applies.
- `LLAMA_DSPARK_EMPTY_COOLDOWN`: `0` (default) or `1`. After confidence admission
  produces no usable draft, skip drafting at the next logical token position.
  A rewind, request reset, or intervening plain target step expires the hint.

These controls do not change the denoising block width or target verification.
Cooldown state is per sequence and is included in speculative checkpoints.
`LLAMA_DSPARK_PROFILE=1` logs selections and skipped attempts. Neither control
changes DFlash, MTP, or other speculative implementations.

## Experimental verification KV packing

KV packing for six target rows (anchor plus five drafts) and CUDA shared-KV
concat fusion for five or six streams are enabled by default.
`LLAMA_DSV41_VERIFY_KV_WIDE=0` or `off` restores the previous limits.
Set this variable before starting the process.
The existing single-sequence, causal, flash-attention, F16 KV, and
`LLAMA_DSV41_VERIFY_BATCH` guards still apply. Unequal per-query KV lengths
use the existing fallback.

This experiment keeps the floating and quantized matmul guards unchanged.
It does not extend compact attention beyond its current width or change the
24576-compressed-record threshold, draft policy, or denoising block width.

## Experimental quantized verification samples

Eligible CUDA-resident quantized dense projections are batched as independent
one-column MMVQ samples by default. The
single-column reduction is retained for five or six target rows. The existing
short-verification guards apply; six rows also require KV_WIDE. Set the variable
before starting the process. `LLAMA_DSV41_VERIFY_QUANT_SAMPLES=0` or `off`
restores the previous path; unset, `1`, or `on` enables it. CPU weight tensors
keep their existing fallback.

This path leaves MoE dispatch and floating matmul unchanged. It applies only
when verification reaches five or six rows (anchor plus drafts), so draft limits
of three or fewer do not use it. Full-Q3 fixed-prefix/rollback checks and
Ada/Ampere operator checks passed. Text/code512/8K tests showed +3.24% geomean
TG, with code gains of +6.80%/+5.41%; long-context and broad quality gates remain
open. The default was enabled after separate user review of the code gains;
the default draft policy is unchanged.

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


## Automatic MoE prefill staging

`LLAMA_MOE_PREFILL_STAGE=auto` enables startup calibration for the existing Q3_K expert staging path. An expert cache must be enabled (`--moe-expert-cache` or the MiB/device interface). Auto remains opt-in and does not change tensor placement or speculative policy.

The runtime measures CPU versus the actual pinned/double-buffer staged FFN on eligible CUDA devices, with full512-token chunks and the configured batch CPU threads. It uses isolated synthetic FFN inputs/rotating expert IDs, warmup2+three trials, without changing KV state or expert LRU contents. A device needs workspace/cache headroom plus1024MiB reserve and at least5% measured gain. When CPU wins, auto explicitly pins host projections to CPU; the stock scheduler can otherwise offload them to CUDA. Unsupported layers retain stock scheduling. Auto leaves shorter prefill tails on CPU; manual `=1` and `LLAMA_MOE_PREFILL_STAGE_DEVICE=CUDAx` retain their existing behavior.

Costs are saved next to the first GGUF shard as `<first-shard>.moe-prefill-costs.json`. `LLAMA_MOE_PREFILL_STAGE_COSTS` optionally selects another file. Matching model/runtime/CPU/GPU identities recheck CPU and the best eligible staging candidate at startup; changes or cost drift cause fresh calibration. Atomic publication and a directory lock protect the previous file. Corrupt/foreign files or an unwritable model directory are reported and preserved; calibration can continue in memory. Metadata-fit only reserves possible workspace and never runs calibration or writes files. See [the calibration tool documentation](../tools/moe-cache-calibrate/README.md) for identity, numerical checks and limitations.


## Frozen MoE route profile

With `--moe-expert-cache-profile FILE`, common startup holds one locked input snapshot across memory fitting and real initialization. Metadata contexts validate the schema, counts, metadata and tensor layout without reading weight payloads or saving the file. Real initialization also validates sampled weights. Metadata without an explicit snapshot retains the previous no-profile-I/O behavior. Target and draft profile paths remain separate.

The real profile keeps mutable observed counts separately from the immutable startup input and saves them at model teardown. A changed source file is preserved with a save error instead of being overwritten. The profile lock remains held until its last owner is destroyed. Slots, cache capacity, eviction and production defaults are unchanged. The experimental context parameters gained an optional snapshot pointer; applications using this branch must rebuild with the matching headers. Internal C++ callers can hold a snapshot from `llama_moe_profile_snapshot_load` in `llama-ext.h`; `common_init_result` prepares it automatically.


## Profile-based MoE cache capacities

`LLAMA_MOE_CACHE_CAPACITY_PROFILE=1` enables experimental per-layer capacities with `--moe-expert-cache-mib N`, `--moe-expert-cache-device CUDAx` and `--moe-expert-cache-profile FILE`. Common startup supplies the frozen profile automatically; direct C++ callers must pass it to budget initialization. The default and legacy `--moe-expert-cache N` remain unchanged. A missing or zero-heat profile uses uniform capacities.

The planner starts with one slot per eligible layer. It then adds slots by marginal route share per aligned host-bundle byte, with a 10% uniform prior and per-layer normalization of observed heat. Equal scores prefer the lower occupied expert fraction, then layer ID. This is a greedy capacity heuristic, not a global optimum or a latency guarantee. Each layer retains its slots and the existing LRU. Dummy slots, device tables, alignment and same-device staging are included in the cap. Automatic staging reserves its possible home workspace and temporary storage for both metadata and real planning even when CPU wins, so capacities do not depend on noisy calibration choices. Real allocation can use less than this conservative plan.

The user still sets the total MiB cap and device. This feature does not automatically choose a VRAM budget, move compute to a different GPU, share slots dynamically, or introduce spare-slot replacement. Slots are frozen for the live owner. Fully resident projections stay resident; authoritative host weights are retained.
