# DeepSeek41 floating sample-grid6, 2026-11-06

Status: primitive exactness and the paired fixed-prefix width screen PASS. No complete lifecycle/two-corpus/native speed/default promotion claim yet.

After db4be119e, MoE5+1 stays explicit1 in every full-model arm, prefill cache deferral0. The normal default-ON build completed before this prototype: production libllama e23f1272, libggml-base37791769, libggml-cuda6329797c. Earlier old-module timing is not a reference for the new comparison.

## Bounded graph change

`LLAMA_DSV41_VERIFY_FLOAT_WIDE=1` or `on` opts into the existing four-dimensional sample view at six target queries. Unset/0/off stays OFF. Existing FLOAT_BATCH must also be enabled. Keep six-query/n_rs_seq5/text/P1/causal flash/KQV/pipeline/no callback/no warmup guards, Q8 CSA+LID and a CUDA weight buffer. Existing floating type/even K/contiguous weight/F32 input/stride/channel guards remain. Check backend support before returning the wide operation. Grouped wo_a, dynamic or CPU weight buffers and unsupported layouts fall back to the scalar path. The global five-query predicate and every CUDA source/selector remain unchanged; shared-weight MMVF still caps at5.

Only src/llama-graph.cpp is changed for this experiment. Private library SHA256 `3bb55e36ca81fa9008d5348d44f3f5fbfa28951971b75b20388e0f8dfa318690`; production module is unchanged by Phase B. [Patch](artifacts/runtime.patch), [build](artifacts/build.json), [protocol](PROTOCOL.md).

## Primitive gate

The existing external MoE operator fixture's backend/allocation/hash scaffold was adapted to compare six-sample GEMV with six independent F32-precision columns. All360 evaluations pass bitwise, max_abs0: F32/F16/BF16 weights, logical CUDA0 Ada and CUDA5 Ampere, K/M6144/384,6144/512,1536/8192,512/128,128/64, channels1/2/64, packed versus padded input strides, two different input buffers. These representative synthetic matrices do not replace the actual full-model tensor/state gate. Operator process exit0, sampled free minimum23713.25MiB, no memory-floor breach. [Source/build](artifacts/operator-build.json), [results](artifacts/operator-results.json), [controller/memory](artifacts/operator-run.json).

The first adapted controller retained its old R-relative output directory and ran the old64-case MoE fixture, then failed its expected360 assertion. That invocation did not exercise floating6. Its files are retained under [runner-path-error](artifacts/runner-path-error.json); the three overwritten historical artifacts were restored byte-for-byte from HEAD before the corrected run. The corrected runner resolves its own directory and asserts the expected experiment name.

## Full-model gate and remaining work

[Fresh joint admission](artifacts/admission.json) retains15360GPU expert bundles/identical routes and a conservative1267.25MiB floor after prior actual overhead plus64MiB guard. This estimate does not replace monitored allocations. The paired width controller reuses the byte-identical existing harness, frozen inputs, ctx92160/Q3/Q8/K5/TopK2048/B8192/U512/P1/threads12, no callback/CUPTI/interposer. It checks every raw output hash and sampled state, with actual free>=512MiB and clean exits.

Full two-corpus logits/taps/state/restore/rollback/failure-retry and clean native OFF/ON/OFF remain required. Also measure Phase B alone with MoE0 separately from the combined MoE1 cohort. No source/default promotion or deployment follows from a primitive/metadata pass. Old uninstrumented prefill/rollback failures remain unresolved. Phase C shared-weight6 CUDA changes are excluded.


## Completed paired target-width screen

Same private module in OFF/ON, only FLOAT_WIDE0/1 changes; MoE_WIDE1 in both. Both fresh processes use the newly rebuilt CUDA/base modules. Each prefix has five measured repetitions per width per forward/reverse pass, plus retained warmups. Target decode+sync only; snapshot restore, bridge/draft, raw output transport and hashing are outside the timer. No callback/CUPTI/interposer. All observations remain, including cold/variable rows. This is a raw-logit target screen, not server GPU-sampler TG.

| Prefix tokens | Target queries | OFF median ms | ON median ms | Latency change |
| --- | --- | --- | --- | --- |
| 8192 | 1 | 24.242 | 24.069 | -0.71% |
| 8192 | 2 | 30.217 | 30.205 | -0.04% |
| 8192 | 3 | 34.862 | 34.834 | -0.08% |
| 8192 | 4 | 39.036 | 39.090 | +0.14% |
| 8192 | 5 | 43.113 | 43.109 | -0.01% |
| 8192 | 6 | 54.654 | 52.772 | -3.44% |
| 90112 | 1 | 24.401 | 24.339 | -0.25% |
| 90112 | 2 | 31.851 | 31.793 | -0.18% |
| 90112 | 3 | 36.471 | 36.470 | -0.00% |
| 90112 | 4 | 42.294 | 42.302 | +0.02% |
| 90112 | 5 | 46.103 | 46.052 | -0.11% |
| 90112 | 6 | 57.357 | 55.561 | -3.13% |

All156 raw observations per arm and24 sampled target-state checkpoints per arm match across flags, including base state/proposal IDs and first-row versus width1. Both processes exit0 with no floor breach, forced kill or CUDA/OOM/compute-buffer expectation warnings. Actual sampled minima: off 1395.125MiB, on 1401.125MiB. All measured frontend graphs are reused. Scheduler split counts remain58 at8K and57 at88K in both arms; actual floating-node/CUDA-launch counts still need the separate diagnostic. No fresh CUDA-instantiation counter was added to this timing. [Raw manifest](artifacts/width-results.json), [OFF rows](artifacts/width-off/results.json), [ON rows](artifacts/width-on/results.json), [summary](artifacts/width-summary.json).

The phase is still opt-in/OFF. This screen exercises both widths and state restores, but it does not replace the full rejected-window rollback/taps/two-corpus/fault matrix, native PP/TG, or the MoE0-only cohort. A lower target-call median is not an end-to-end throughput claim. Next: confirm the actual floating-node/dispatch reduction separately, then full lifecycle and native controls. No Phase C shared-weight selector change or new commit is included.
