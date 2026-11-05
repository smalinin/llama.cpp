from pathlib import Path
import json,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105';P=E/'artifacts';j=json.loads((P/'native-summary.json').read_text());assert j['complete'];b=json.loads((P/'build.json').read_text());rows=j['rows'];s=json.loads((P/'STATE.json').read_text())
table='\n'.join(f"| {r['prefix']} | {r['pp']['off_mean']:.2f} | {r['pp']['on_mean']:.2f} | {r['pp']['delta_percent']:+.2f}% | {r['tg']['off_mean']:.2f} | {r['tg']['on_mean']:.2f} | {r['tg']['delta_percent']:+.2f}% |"for r in rows)
ranges='\n'.join(f"| {r['prefix']} | {r['pp']['off_range'][0]:.2f}..{r['pp']['off_range'][1]:.2f} | {r['pp']['on_range'][0]:.2f}..{r['pp']['on_range'][1]:.2f} | {r['pp']['control_drift_percent']:+.2f}% |"for r in rows)
minima='\n'.join(f"| {label} | {min(phases['load'].values()):.2f} | {min(phases['inference'].values()):.2f} |"for label,phases in j['sampled_min_free_mib'].items())
text=f'''# DeepSeek41 prefill cache deferral, 2026-11-05

Status: approved prototype implemented and validated in the bounded configuration below. Default OFF. It improves measured PP8K by4.45% and PP88K by7.40%, but regresses PP512 by4.74% and uses more joint target/draft/tap workspace on CUDA5. It remains opt-in for review. No new commit, default promotion or production deployment was performed in this implementation stage.

## What changed

The graph reuses existing remote CSA/LID cache_updates and SET_ROWS operations. Eligible text prefill ubatches defer remote replica writes until the first local reader. KV-only early return explicitly expands every remaining write and clears the map before the existing n_outputs0/empty-map assertion. Local and unrecognized-owner writes stay eager. Existing decode selection/ordering is retained.

Initial scope: DeepSeek41 token input,2..512tokens per ubatch, one sequence/slot, causal flash attention, layer split, full GPU attention/FFN/routed/shared weights, pipeline enabled, Q8_0 CSA/LID storage and no eval callback. CPU Engram tables remain supported. Embedding/media input, parallel slots, unsupported cache types/backends and partial offload use the original prefill path. Source graph guards are separate from the narrower measured cohort below; they do not establish general model support.

No new arithmetic, quantization, input/output pointers, scheduler events, protected copy slots or cross-graph pending jobs. A nonvirtual forwarding cache type accessor avoids adding graph nodes just to inspect storage. The private module was built from two changed CXX objects and existing unchanged build objects; class layouts did not change.

[Approved design](../../designs/1017-prefill-cache-deferral.md), [preceding owner audit](../1017-kv-owner-audit-20261105/REPORT.md), [exact runtime patch](artifacts/runtime.patch), [build commands and identities](artifacts/build.json). The initial implement.py edit trace predates guard refinements; runtime.patch and final source snapshots define the built code.

## Frozen cohort

New host: Threadripper PRO3945WX,12CPU threads,192GB RAM and the recorded GPU UUID/PCI order. Source baseline f796e544f, documentation HEAD cf1e5a461. Fixed Q3_K-AntirezLike target with MXFP8 Engram, Q8_0 K/V, TopK2048, B8192/U512/P1, all15360 routed expert bundles on GPU, fixed tensor overrides, expert cache0 and prefill staging disabled. No mmproj/auto-fit. CPU lookup early remains unset/default ON for every arm.

DSpark: same BF16-named file as the baseline, actual mixed weights (mostly MXFP4), K5/full/pmin0.5, draft CUDA5/CUDA0 and the same layer1 down override. The filename is not a claim that all draft tensors are BF16. Exact argv/environment, model stat identities and library/source hashes are recorded in [native manifest](artifacts/native-results.json). Models and presets were not modified.

Ctx100352 ON failed conservative512MiB admission at464.23MiB on CUDA5. Ctx99328/98304/94208 also failed. Per the user's prior authorization to reduce context, both OFF and ON use92160. Long prefix90112 is88K, not98K. The chosen conservative bound is519.86MiB after previously measured per-device overhead plus64MiB guard. No speed comparison to earlier ctx100352 cohorts or old-platform measurements is made.

## Clean native speed

Three fresh processes OFF/ON/OFF using the same private module. Every prefix has one warmup plus two measured repetitions in each process. The table compares ON's two measured repetitions with the four measured OFF repetitions bracketing it; warmups are excluded. Each request has a full uncached prompt and256greedy output tokens. PP uses prefix length/prompt time; TG uses255intervals/predicted time. No interposer, profiler, eval callback or diagnostic observer in this speed cohort. The external memory monitor is identical in all arms.

| Prefix tokens | OFF PP t/s | ON PP t/s | PP change | OFF TG t/s | ON TG t/s | TG change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

| Prefix tokens | OFF measured PP range | ON measured PP range | OFF-post vs OFF-pre mean |
| --- | ---: | ---: | ---: |
{ranges}

The medium/long PP gains exceed the measured control drift, while the PP512 regression also repeats across controls. Stable TG acceleration is not established: its ON repetition spread overlaps or approaches the controls; this is a short screen, not a statistical confidence interval or a long-generation quality comparison. No fresh matched Maya run at this smaller context was performed.

All27response token arrays and draft generated/accepted counters match exactly for each prefix, including warmups. Every prompt reports cache_n0 and the full requested prompt length; each response contains256tokens. All three processes exit0 without forced termination, allocation-size mismatch, OOM, CUDA errors or512MiB sampled-floor breach. [Summary with per-arm means/ranges and memory](artifacts/native-summary.json), [controller](artifacts/native.py), [run log](artifacts/native-run.log).

## Precision and lifecycle

Actual flags0/1 used the same private module and context. Twelve paired result groups (six per corpus;24records across both arms) match exactly on two88K prose/code corpora: full target/draft logits, exported taps, compressor/ring/KV/state hashes, full/partial/sequence restore and replay, K3/K5 drafts and six rejected-window rollback variants per corpus. Logical appended widths1/7/508/511/512/513 include the513split512+1 and copy-slot reuse.

Teacher-forced NLL uses128tokens per corpus: prose0.37295498784483416, code0.1052687732276846, delta0. Two seed-fault/retries per arm (four total) pass; each arm records780calls,176injection graphs and two injected failures. Backend sampling is disabled for raw logits; there is no eval callback, only the existing small fault interposer. These exactness checks are appropriate because scheduling changes, not arithmetic.

Actual precision sampled minimum free VRAM: OFF1387.125MiB, ON801.25MiB, both exit0 without warnings. These raw checks are separate from native backend sampling/actual allocation above. [Precision manifest](artifacts/precision-results.json), [harness](artifacts/precision.cpp), [controller](artifacts/precision.py).

The existing scheduler forwarding/protected-slot/CPU-lookup/failure fixture is rerun after native timing. It validates the unchanged ggml-base module, not the DeepSeek model ordering itself; full-model precision and ownership checks supply that evidence. Results and fixture provenance are in artifacts/scheduler-results.json and artifacts/scheduler-rerun.json.

## Actual ownership and graph structure

Ten paired no_alloc graph variants pass symbolic operator and SET_ROWS operand/count checks, including full-tail and KV-only511/512shapes. Node counts remain equal; ON labels seven deferred writes. Constants, named operands and value expressions remain in the fingerprint. Structural checks do not replace numeric/state evidence.

Actual scheduled metadata confirms source20 transformed values now originate on CUDA2 (OFF:CUDA4), without a separate transform-placement patch. All36source20 CSA writes remain across nine captured target chunks, with four remote cache allocations and writes before their first readers. Four selected KV-only chunks flush all pending remote replicas. Executed scheduler graphs change31->24 for KV-only and50->40 for full-tail chunks; these counts are not a PP speedup ratio.

An additional comparison covers all419raw/CSA/LID SET_ROWS operations per arm across the same nine chunks (four with36writes, five with55). Root types, sizes, destination backends, value/index shapes and counts match; every distinct persistent allocation is written once per selected ubatch, before any first actual reader present in that graph. Raw ring, CSA and LID reuse cache_k_lN labels, so root names alone are not allocation identity. The analyzer uses actual root data/storage size within a ubatch, not pointer equality across processes. All ten diagnostic16token outputs and draft counters match OFF/ON.

The observer only forwards existing graph/copy/event operations and reads metadata; no numeric device data, new GPU events or CUDA API hooks. Its armed PP timings are perturbed and excluded from the native speed table. Existing protected copy/event paths remain intact. [Actual ON proof](artifacts/owner-on/owner-checks.json), [all cache writes](artifacts/all-cache-writes-checks.json), [same-context default capture](artifacts/owner-default/summary.json). Compressed raw traces have byte/hash round-trip checks in each owner folder's payload.json.

## Memory cost and context limit

Weights, model context and routed placement are unchanged between arms at92160. The metadata budget reports CUDA5 joint allocation +795.02MiB: main compute -4MiB, draft/taps context +160.02MiB and draft/taps compute +639MiB. CUDA4 falls164.02MiB, CUDA1/2 each4.25MiB; aggregate increase622.5MiB. These are existing allocator budget categories, not newly hand-managed buffers or proof of an isolated root cause. Cache write deferral changes live ranges/placement in the joint graph. A workspace reduction would be a separate candidate.

| Native arm | Sampled minimum free during load, MiB | Sampled minimum free during inference, MiB |
| --- | ---: | ---: |
{minima}

ON actual owner trace minimum587.25MiB; native long-prefix inference reaches535.25MiB, only23.25MiB above the512MiB requirement. Metadata estimates alone would miss this distinction. Current admission is for this fixed placement/context and does not prove262K joint auto-fit, F16 KV, media, parallel>1, CPU expert caching/staging or another model.

## Diagnostic corrections retained

- The first reserve observer was oververbose and deliberately stopped. Its initial source/log/raw trace locator/hash remain; the bounded observer records small headers and at most12unique full graph shapes. The11GB raw file is ignored locally, not proposed as a versioned payload.
- The symbolic checker initially treated CPY's destination self-link as a value cycle and anonymous FILL shape scaffolding as different values. Only destination/shape-only generated labels were normalized; named operands/constants/value expressions remain. Initial diagnostics and the corrected checker are retained. Runtime source did not change for these corrections.
- The first intended ON ownership controller omitted the flag due to a generator insertion error. Activation checks caught actual default OFF before native timing began. That manifest remains byte-identical under owner-default; owner-default-recovery.json records the mistake. Precision explicitly carried0/1 and is unaffected. The corrected controller explicitly inserts/asserts1; actual deferred labels and physical producers prove activation. No runtime source/binary changed to fix the controller.
- The all-cache analyzer initially keyed by cache_k_lN and rejected distinct raw/CSA/LID allocations sharing that label. Actual widths128/512, storage sizes and root data resolve the identity; initial/debug/final scripts and cache-root-label-check-notes.json are preserved. This was a checker error, not a duplicate runtime write.

## Current decision and follow-up

Keep the prototype opt-in/default OFF for review: the PP8K/88K benefit is useful, but PP512 regresses and CUDA5 memory pressure is materially higher. No scheduler fence removal or default promotion follows from these results. A separate candidate could reduce the joint target/draft/tap workspace and investigate the short-prompt regression before widening deployment. The previously planned six-query float/MoE verification work remains independent.

After building this source, LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=1 enables the guarded mode at context creation; unset/0 retains eager prefill. Treat it as a startup setting. Existing LLAMA_DSV41_DEFER_CACHE_UPDATES retains its decode default/conditions. Production modules and user presets still use their previous implementation.

Private libllama SHA256: {b['private_llama_sha256']}.
Production libllama SHA256 remains {b['production_llama_sha256']}.
Production ggml-base SHA256 remains {b['production_base_sha256']}.
'''
(E/'REPORT.md').write_text(text)
s['complete']=True;s['stage']='Implementation, structural/actual ownership, full precision/lifecycle, and clean native OFF/ON/OFF complete. Default OFF; review trade-offs.';s['native_summary']='native-summary.json';s['next']=['Review PP gains vs PP512 regression and CUDA5 workspace cost; no default promotion/deploy/commit without request.','If approved, isolate joint workspace/short-prompt improvements; six-query verification stays a separate planned candidate.'];s['native_min_free_mib']=535.25;s['all_native_outputs_counters_exact']=True;s['source20_owner_off_on']=['CUDA4','CUDA2'];s['all_cache_writes_checked_per_arm']=419;s['sessions']={'owner_on':'completed','precision':'completed','native':'completed','scheduler':'see scheduler-rerun.json'};(P/'STATE.json').write_text(json.dumps(s,indent=2)+'\n')
shutil.copy2('/tmp/finish-prefill-deferral-report.py',P/'finish-report.py')
for f in ['correct-deferral-owner-controller.py','run-deferral-native-after-owner.py']:
 if Path('/tmp',f).exists():shutil.copy2(Path('/tmp',f),P/f)
print('REPORT_WRITTEN',len(text))
