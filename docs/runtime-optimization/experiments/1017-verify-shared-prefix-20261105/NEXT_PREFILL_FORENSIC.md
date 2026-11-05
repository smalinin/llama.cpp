# Next: capture full-prefill variability before changing runtime

Reuse the original raw precision harness setup with ctx92160/Q8/Q3/DSparkK5 and the original unarmed fault interposer, MoE-wide0 and prefill cache deferral0. No eval callbacks or new GPU reads before completion of the first8K target+draft apply. The new six-query branch cannot activate during that prefix.

Retain the existing first8K capture order: taps37/38/39, target sequence/full, draft sequence/full. Preserve those host bytes instead of discarding them after hashing. Only after every original read/hash completes, write content-addressed raw blobs and compare the complete group. Extra forensic CPU/disk work occurs after the candidate failure point. Do not silently normalize uninitialized state, NaNs, signed zeros or metadata.

First fresh process captures the reference without choosing a seed by its canonical hash. Record whether it matches the archived group. Run three clear/reset repeats per process, then fresh startup processes with the same frozen identities. Localize first changed tap/serializer component and quantify raw differences. A passed same-process repeat does not replace a fresh-start control. Retain failures and do not call a valid numerical backend difference a hardware fault without evidence.

After a reproducible divergence is captured, isolate one supported scheduling/graph/numerical mechanism at a time. No new runtime ownership/order change follows from this protocol; that needs a separate reviewable design if required. The current source already uses per-device/per-stream cuBLAS handles and explicit workspace; do not assume a shared cuBLAS handle is the cause. Do not remove scheduler fences or relax exactness based on hash-only symptoms.

Once prefill variability is understood or controlled, rerun full two-corpus/NLL/fault gates for5+1, followed by native same-K OFF/ON/OFF speed. Phase B/C remain separate candidates, not combined into the forensic cohort.
