---
name: runtime-optimize
description: Profile and optimize local llama.cpp inference using the versioned runtime knowledge base, reusable experiment evidence, and model-specific correctness and speed gates. Use for PP/TG, KV, MoE, pipeline, or speculative-decoding optimization work.
---

# Optimize llama.cpp runtime with recorded evidence

Read the repository instructions and [knowledge index](../../docs/runtime-optimization/README.md). Follow the user's current task; documentation does not authorize benchmarks on occupied GPUs, model changes, default promotion, commits, or deployment.

For a new model, use the [HowTo](../../docs/runtime-optimization/HOWTO.md) and [model passport template](../../docs/runtime-optimization/templates/MODEL.md). Establish actual tensor types, KV/source topology, draft consumers, hardware, placement and binary identity before applying an optimization. DeepSeek constants and previous thresholds are model-specific evidence, not presets for another model.

For a known bottleneck, read only the relevant [catalog entries](../../docs/runtime-optimization/CATALOG.md), [lessons](../../docs/runtime-optimization/LESSONS.md) and linked local evidence. Check implementation and current defaults in the checked-out source; archived reports retain their historical status. Distinguish missing evidence from a failed gate.

Freeze input IDs, expanded argv/environment, placement, KV and sampling. Change one mechanism; confirm the candidate branch actually runs. Callback-based profiling may disable optimized paths. Separate profiler diagnostics from clean native A/B and measure complete costs, including transfers, dispatch and synchronization.

Preserve explicit producer/consumer ownership and required masks, rounding, source states and rollback. Choose exactness or numerical-quality gates according to the arithmetic change. A greedy match, operator gain or successful no_alloc estimate does not replace logits/state/lifecycle, actual memory and native speed evidence.

Use the [experiment template](../../docs/runtime-optimization/templates/EXPERIMENT.md) to record results and scope. Preserve failed experiments and measured source/binary hashes; update the catalog/passport when conclusions change. The local [DS41 passport](../../docs/runtime-optimization/models/deepseek41-1017.md) is a worked example. Historical runners in [evidence](../../docs/runtime-optimization/evidence/README.md) need path remapping and matching internal headers before execution.
