# GLM-5.3 Implementation Plan

## Scope

This plan covers the next two development tracks in
`/home/sergei/Github/llama.cpp` on branch `my_build_glm53_flash`:

1. Adapt the useful GLM-DSA/MTP improvements to GLM-5.3-Flash (`glm5_next`).
2. Port the GLM-5.2 `glm-dsa` fixes to standard GLM-5.3 (`glm_moe_dsa`).

The target branch starts this work at `cc6dea3b2`. The existing GLM5NEXT work
was originally based on Unsloth's
[`glm5next/upstream`](https://github.com/unslothai/llama.cpp/tree/glm5next/upstream)
branch. The reference GLM-DSA fixes are available in
`origin/my_build_glm52`, ending at `0b5c2f91a`.

## Architecture boundary

| Model | llama.cpp architecture | Indexer layout | Relevant implementation |
| --- | --- | --- | --- |
| GLM-5.3-Flash | `LLM_ARCH_GLM5NEXT` / `glm5_next` | Full indexer in every sparse layer | K-pool cache and custom indexed FlashAttention |
| GLM-5.3 | `LLM_ARCH_GLM_DSA` / `glm_moe_dsa` | Full and shared indexer groups | DSA cache and generic sparse FlashAttention |

The two paths must remain separate. In particular, shared-indexer mask reuse is
useful for standard GLM-5.3 but does not apply to GLM-5.3-Flash.

## Working rules

- Adapt changes to the current APIs; do not blindly cherry-pick old commits.
- Preserve CPU and unsupported-backend fallbacks.
- Add an A/B switch for a new fast path whenever a safe fallback is practical.
- Each implementation stage gets its own code commit.
- After each code commit, update `GLM5NEXT_LOCAL_CHANGES_EN.md` in a separate
  documentation commit, then stop for review before starting the next stage.
- Do not include merge commits from `origin/my_build_glm52`.
- Record correctness and performance results using the same prompts, sampling
  seed, context, cache types, slot count, and draft parameters.

## Phase 0: Freeze the baseline

This phase changes no production code.

- Build the current CUDA configuration and record the build flags and GPU model.
- Run the existing backend, architecture, K-pool, speculative, and server tests.
- Record GLM-5.3-Flash results with MTP disabled and enabled at short context,
  50K, 80K, and 100K context.
- Record prefill speed, generation speed, draft acceptance, mean accepted length,
  VRAM use, and CUDA graph reuse for `n_slots = 1` and `n_slots = 2`.
- Keep the raw baseline logs outside the repository unless a compact benchmark
  summary is worth committing.

Exit criterion: a reproducible baseline exists and the current test suite passes.

## Phase 1: Useful adaptations for GLM-5.3-Flash

### F1. Add regression coverage before changing the fast path

Reference ideas:

- `8b38bc580` - sequential GLM MTP drafting coverage.
- `ab345947d` - multi-stream sparse FlashAttention coverage.

Adapt the tests to `glm5_next` and its custom `fattn-indexed` operation:

- Verify sequential MTP steps, reuse enable/disable transitions, accepted-token
  catch-up, rejection, rewind, and a fresh request after slot reuse.
- Exercise indexed attention with at least two streams/sequence IDs, including
  different valid KV lengths and invalid/padded indices.
- Cover both F16 and Q8_0 KV-cache paths where supported.

Commit: `tests: cover GLM5NEXT sequential MTP and multi-stream indexed attention`

Exit criterion: tests pass under CUDA compute-sanitizer where practical and do
not change model output.

### F2. Skip redundant indexer key and gate work during MTP reuse

Reference idea: `0fd68f6c6` (`glm-dsa: skip reused MTP indexer keys`).

The current GLM5NEXT graph reuses top-k after the first MTP iteration, but still
computes and writes the indexer key and compressor gate. Move the reuse decision
before this work when the cached selection is valid.

Requirements:

- Keep the incomplete tail and newly accepted positions correct.
- Mark modified pools dirty and rebuild them during target catch-up.
- Isolate cache state per stream/slot.
- Reset reuse on rejection, rewind, sequence reset, prompt reprocessing, and
  context shift.
- Retain `LLAMA_GLM5_MTP_TOPK_SHARE=0` as the A/B/correctness fallback.

Commit: `glm5next: skip redundant MTP indexer key updates`

Exit criterion: identical greedy output and acceptance decisions with reuse on
and off; lower MTP graph time without regressions for multiple slots.

### F3. Keep the complete MTP inner loop on the backend

Reference idea: `f09e46594` (`glm-dsa: keep MTP draft state on backend`).

GLM5NEXT already keeps hidden-state transfer on the GPU, but each MTP step still
synchronizes the draft context and feeds the sampled token back through the CPU.
Adapt the device-resident draft loop to the existing GLM5NEXT multi-slot hidden
buffers and adaptive draft limits.

Requirements:

- Keep sampled token/probability state and the next hidden-state input on the
  backend between MTP iterations.
- Read back only the compact final result required by speculative verification.
- Preserve `spec-draft-n-max`, `n_min`, `p_min`, and remaining-context limits.
- Support independent per-sequence ranges for `n_slots > 1`.
- Keep the current host-driven path as a runtime fallback.
- Do not enable the path for unsupported samplers, multiple NextN heads, or
  incompatible shared-memory/chain-head configurations.
- Preserve the multimodal GPU-direct embedding path.

Commit: `glm5next: keep MTP draft iterations on backend`

Exit criterion: no per-step host synchronization in the supported path; matching
tokens and acceptance versus the fallback; measurable decode improvement at
short and long context for one and multiple slots.

### F4. GLM5NEXT validation checkpoint

- Run the Phase 0 matrix again.
- Add multimodal prompt reuse and multimodal generation cases.
- Check cancellation, slot reuse, LCP rewind, context shift, and auto-fit/OOM
  boundaries.
- Compare MTP on/off and every new A/B switch.

Checkpoint status: in progress. Validation exposed and fixed a backend-resident
MTP context-shift failure in `803b76c92`. The corrected 7800-token prompt plus
512-token generation crossed an 8192-token context boundary with identical
device/host-fallback output and acceptance. Remaining long-context/cache-type
matrix items still require review and execution before F4 is approved.

No optimization work starts in Phase 2 until this checkpoint is reviewed.

## Phase 2: Port the `glm-dsa` fixes for standard GLM-5.3

The official GLM-5.3 configuration uses `GlmMoeDsaForCausalLM` with
`model_type = glm_moe_dsa`, one NextN layer, `index_topk = 2048`, and both full
and shared indexer groups. The port therefore targets `src/models/glm-dsa.cpp`,
not `src/models/glm5next.cpp`.

### D0. Structural audit against the current branch

Classify each hunk from the reference series as `already present`, `superseded`,
`missing`, or `not applicable` before editing.

Important known results:

- The fused-QKV `TENSOR_SKIP` fix from `a569338e3` is already present.
- The old provisional server fit from `ec699a26a` is superseded by
  `common_fit_extra_model`, which fits target and MTP/draft contexts together.
- The old NextN layer-count changes (`a569338e3`, `7f39659e6`, `204e32d97`)
  must not be copied over the newer fit logic without an OOM regression test.
- `b028e9623` radix top-k is deferred: current CUDA top-k has no observed
  49K-to-65K cliff, and GLM-5.3 uses `topk = 2048`, outside that old fast path.

Exit criterion: a hunk-level transfer table is ready. This audit alone requires
no commit unless documentation changes.

### D1. Port the GLM-DSA correctness foundation

Reference: `b6fa6a70b`.

Adapt only the portions still missing from the current branch:

- Correct dense-attention Q/K/V inputs.
- Correct sparse-versus-dense dispatch.
- Preserve sequence-aware KV mapping and bounds in sparse FlashAttention.
- Bring the GLM-DSA graph and cache interfaces to their final current-API form.
- Add focused architecture/backend tests for every transferred correction.

Commit: `glm-dsa: fix attention inputs and sparse dispatch for GLM-5.3`

Exit criterion: dense and sparse reference comparisons pass for single- and
multi-stream batches without changing unaffected architectures.

### D2. Add safe MTP top-k reuse and require a full NextN indexer

References:

- `e9595867d` - reuse top-k across MTP steps.
- `f5b08c276` - require a full MTP indexer.
- `8b38bc580` - sequential MTP regression coverage.

Implement the final corrected behavior directly:

- Convert and load the indexer-type metadata needed to distinguish full and
  shared indexers.
- Allow MTP top-k reuse only when the NextN indexer is full and compatible with
  the target selection.
- Reset reuse at all sequence-state boundaries.
- Add sequential exactness tests before enabling the optimization by default.

Commit: `glm-dsa: safely reuse full-indexer top-k during MTP`

Exit criterion: MTP output and acceptance match reuse-disabled execution across
accept/reject/catch-up sequences.

### D3. Keep the GLM-DSA MTP top-k cache on the backend

Reference: `aa96a2059`.

- Store top-k state in backend buffers.
- Keep cache ranges independent per sequence/slot.
- Avoid full top-k CPU round trips.
- Preserve a host fallback for unsupported backends.

Commit: `glm-dsa: keep MTP top-k cache on backend`

Exit criterion: no host copy of the full top-k array on the supported CUDA path;
single- and multi-slot tests pass.

### D4. Skip reused GLM-DSA MTP indexer keys

Reference: `0fd68f6c6`.

- Skip indexer-key work only while the cached MTP selection remains valid.
- Recompute affected entries after acceptance, rejection, rewind, or cache
  mutation.
- Cover final partial groups and context-boundary behavior.

Commit: `glm-dsa: skip reused MTP indexer keys`

Exit criterion: identical results to the non-reuse path with reduced per-draft
graph work.

### D5. Keep GLM-DSA MTP draft state on the backend

Reference: `f09e46594`.

Adapt the same backend-resident loop contract used in Phase F3 rather than
creating a second incompatible implementation.

- Reuse generic speculative infrastructure where both architectures agree.
- Keep architecture-specific graph inputs and cache validation separate.
- Preserve adaptive draft limits, sampler fallbacks, and multi-slot isolation.

Commit: `glm-dsa: keep MTP draft state on backend`

Exit criterion: matching fallback output and acceptance, no supported-path
per-step host synchronization, and improved GLM-5.3 decode performance.

### D6. Reuse sparse masks across shared GLM-DSA indexers

Reference: `0b5c2f91a`.

Standard GLM-5.3 contains groups whose first layer has a full indexer and whose
following layers share its selection. Build the sparse mask once per compatible
group and reuse it without changing ownership or lifetime across graph rebuilds.

Commit: `glm-dsa: reuse sparse masks across shared indexers`

Exit criterion: matching logits versus per-layer mask construction, reduced
graph work, and passing tests for full/shared transitions and multiple streams.

### D7. Auto-fit and long-context validation

- Verify that target weights, the NextN layer, target context, and MTP context are
  counted exactly once by `common_fit_extra_model`.
- Test near the VRAM limit with MTP disabled and enabled.
- Confirm that `n_gpu_layers` placement includes all loaded NextN tensors.
- Change fit code only if a reproducible allocation gap remains; make that fix a
  separate commit with a focused regression test.

Potential commit, only if required: `fit: correct GLM-DSA NextN memory accounting`

Exit criterion: no post-fit OOM at the tested safety margin and no unnecessary
VRAM under-allocation.

## Deferred work

- Do not port `b028e9623` radix top-k unless profiling shows a regression on the
  actual deployment CUDA/toolchain and a new implementation benefits
  `topk = 2048`.
- Do not port the generic flash-attention top-k hint contract (`0be18c8ca`) into
  GLM5NEXT indexed attention. Re-evaluate it only if D1 still needs the contract
  for generic GLM-DSA FlashAttention.
- Do not apply shared-indexer mask reuse to GLM-5.3-Flash.
- Avoid duplicating the existing GLM5NEXT backend top-k cache, hidden buffers,
  auto-fit integration, and multi-slot state machinery.

## Final validation matrix

For both model families, run:

- MTP off and on.
- `n_slots = 1` and `n_slots = 2` or more.
- Short context, 50K, 80K, and 100K context.
- F16 and Q8_0 KV cache where supported.
- Greedy deterministic comparison plus the normal production sampler.
- Prompt reuse, LCP rewind, cancellation, sequence reset, and context shift.
- CUDA graph enabled and disabled.
- Auto-fit near the VRAM limit.

For GLM-5.3-Flash, also run multimodal prefill, repeated-image graph reuse, and
generation after multimodal prefill.

Record:

- Prompt tokens/s and per-chunk trend.
- Generated tokens/s and the trend as context grows.
- Draft acceptance and mean accepted length.
- GPU memory usage and allocation margin.
- CUDA graph reuse count.
- Any host synchronization or device-to-host transfer remaining in the hot path.

## Status checklist

- [ ] Phase 0 baseline recorded.
- [x] F1 GLM5NEXT regression coverage committed; awaiting review.
- [x] F2 redundant GLM5NEXT MTP indexer work removed and committed; awaiting review.
- [x] F3 backend-resident GLM5NEXT MTP loop committed and reviewed.
- [ ] F4 GLM5NEXT validation checkpoint in progress; context-shift fix committed and awaiting review.
- [ ] D0 GLM-DSA hunk-level audit completed.
- [ ] D1 GLM-DSA correctness foundation committed and reviewed.
- [ ] D2 safe full-indexer MTP reuse committed and reviewed.
- [ ] D3 backend GLM-DSA MTP top-k cache committed and reviewed.
- [ ] D4 reused GLM-DSA MTP indexer keys skipped, committed, and reviewed.
- [ ] D5 backend-resident GLM-DSA MTP loop committed and reviewed.
- [ ] D6 shared-indexer sparse-mask reuse committed and reviewed.
- [ ] D7 auto-fit and long-context validation completed.
- [ ] Final validation matrix completed and report updated.
