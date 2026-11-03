# Frozen protocol: HC root/reshape sharing

Approved by user2026-11-03 after design review. Runtime base my_maya9b74cc6e6, implementation delta only ggml/src/ggml-backend.cpp. Opt-in GGML_SCHED_COPY_RESHAPE=1, unset/OFF selector retains baseline. No default promotion, commit, model-file or user-preset change is included.

The prototype is narrower than the general design: only a non-inplace RMS_NORM consumer may borrow the HC destination root. A prepass excludes roots with unknown/mutating readers and non-direct/partial/nested views. CUDA-only, n_copies>1, no callback, no INPUT/OUTPUT source/root, canonical destination already in the same split.inputs and protected INPUT/OUTPUT. An allocation-only destination view is added before its first consumer for normal n_views bookkeeping; view metadata is not inserted into the global source-copy table. Root transfer/events/compute_splits/CUDA kernels/allocator policy unchanged.

## Small scheduler gate

External scheduler.cpp reuses the previous pipeline scheduler harness pattern. No tests/* files. CUDA0/CUDA1/CPU, HC_POST64x4xN -> HC_PRE on destination and flat RMS_NORM. OFF/ON/default exact output comparison and actual destination alias/storage/layout assertions.17 variants x3 modes: eligible, single_copy,same_device,late_root,different_split,root_output,root_input,view_output,other_root,other_reader,nested,offset,subrange,strided,cpu_destination,callback,failure. Eligible160 chunks/mode:20 per width1/4/6/7/508/511/512/513. Other variants12 chunks/mode, including one controlled backend compute failure/retry in failure variant. Contexts and host output vectors live until scheduler drain.

Separate fixed-graph32step capture test per OFF/ON proves stable CUDA graph replay and exact changing-input outputs. GGML_CUDA_GRAPH_STATS records capture/launch/update counts. Final counted variant also wraps the owned destination backend's copy interface in the harness to verify two full root/reshape transfer requests per chunk OFF/default and one ON, including a failed compute after its input copies. This wrapper is not production runtime code.

Valid HC_POST and HC_PRE operators require F32 in the public API. F16 HC_POST was not fabricated for a numerical negative test; dtype restriction is checked in implementation/source. Other valid negative layouts/types must not be represented as invalid compute graphs.

## Metadata

metadata.cpp rebuilt with current headers/common/fit.cpp and current libraries. Fixed-layout no_alloc measure100352 default/OFF/ON,176128ON,100352K5/output6 OFF/ON. Target+bound draft/taps budgets, per-tensor routes, all15360 routed GPU bundles, no host experts. Default/OFF parity; ON model/KV unchanged; total combined reserve, not standalone target component, determines projected free. No auto-fit solver proof is inferred.

## Full numerical gate

Existing long.cpp/trace.cpp reused with only selected mechanism changed from M2 selector to copy-sharing selector; M2/default LOCAL stay ON, ASYNC OFF. TargetQ3_K-AntirezLike/MXFP8 Engram/Q8KV/BF16 DSpark,ctx100352/B8192/U512/P1/TopK2048/frozen allGPU split+overrides,fit/cache/staging/mmprojOFF. K5/RS5/output6,p_min0/backend samplingOFF for raw K3/K5 logits/confidences. Two frozen corpora, full98K prefix in each OFF/ON/OFF mode. Target/draft seq/global/partial state, checkpoints at90K/7676+512+4 tail, taps37/38/39, draftK3K5, full/seq/partial restore,replay; rollback4keep1/2/4 and6keep1/3/6, next draft exact; append widths1/7/511/513;128 NLL labels/corpus/mode, expected delta0 and pre-agreed <=0.01nat/token diagnostic bound. Two controlled draft-seed compute failures and full reload/retry.36 result groups,36 rollback workflows. The helper throws on any mode difference; check-long.py validates all raw equality flags, modes, activation and bounded failures.

No clean speed claim from this instrumented/hash/snapshot run. trace.so observes existing scheduler calls without eval callbacks and injects only declared fault cases. NVML100ms, per-GPU cumulative floor512MiB stop guard, load+inference. All helper sources, startup configuration and runtime/library hashes saved. No library rebuilt while GPU jobs run.

## Native cost screen

Only after numerical gate and final small counted gate pass. Three separate server processes OFF-pre/ON/OFF-post using the same frozen context100352/layout, Q8KV/BF16K3/full/p_min0.5/backend samplingON. No preload, profiler, callback, calibration or build activity during timings. Each arm:512warm+2 measured,8Kwarm+3 measured,98Kwarm+1 measured,256 greedy outputs each. Full uncached raw IDs, seed1234,ignore_eos,repeat_penalty1. Compare full arrays and draft counters across all requests, not only timing. Report ranges, medians, OFF drift and ON delta to mean of bracketing OFF medians.98Ksingle measured point per arm and one arm order limit inference; no new full model quality or matched Maya claim.

Watch NVML100ms/stop200ms, per-GPU512MiB floor; stop only owned server. Validate actual memory, unchanged placement/model/KV reserve, exit0 and no OOM/CUDA error/compute expectation mismatch or forced kill. Preserve any raw failure/trace; never rewrite complete/error to disguise a failed run.

After results, write a standalone report and checksums in docs/runtime-optimization/experiments/1017-hc-copy-sharing-20261103, update knowledge/plans and explicitly state kept opt-in/rejected/inconclusive. No automatic default change or commit. The earlier9b74cc6e6 report commit remains separate from this candidate.
