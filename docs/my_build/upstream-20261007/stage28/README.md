# Stage28: GLM5NEXT unified KV pool-cache rebuild

The user requested a fix for the Stage2 concurrent unified-KV abort. Source base: 31aa1e266. V1 is in candidate-bin/source.patch; the final version is in final-bin/source-final.patch. Stop for review before commit. Installed server and presets are not replaced.

The persistent map only contains pools selected by the preceding active sequence set. A returning sequence can have many completed pools still resident in KV but absent from that map. get_max_uncached only sees existing entries; the incremental graph may consequently have fewer rows than set_input discovers.

The shared needs_rebuild overload bounds missing pools for the active sequences using their resident position ranges, subtracting cached entries in those ranges. This includes holes and incomplete boundary pools, so it is a conservative bound. It compares the bound with the existing incremental capacity. If capacity already covers all n_pools, no fallback is needed. Otherwise an oversized bound selects the existing full rebuild graph before allocation. Both graph construction and can_reuse call the same predicate; synthetic reserve overrides are retained. No tensors are resized in set_input, and the capacity assertion stays active.

A second, independently reproduced MTP failure requires disabling top-k reuse in unified multi-slot contexts: the selection storage has one row per stream. These contexts use normal indexer recomputation, while separated and single-slot contexts retain reuse. This can add MTP work even when only one slot is currently active.

The patch does not change slot eviction policy, pooling arithmetic, masks or CUDA kernels. Full rebuild on a returning sequence may cost more than a normal incremental step, especially at long context; throughput is not claimed. Steady decode still uses incremental updates. Entries of inactive sequences are not newly retained, avoiding a separate eviction/lifetime change in this bounded fix.

## Reproduction

Scripts refer to this machine and preserve immutable prior-stage artifacts. Use fresh output directories for reruns. GPU server processes are sequential; CPU checks hide all CUDA devices.

1. `python3 build-candidate.py` rebuilds llama-graph.cpp and llama-kv-cache-kpool.cpp and relinks an isolated libllama. It reuses the Stage27 KV object to retain the committed DeepSeek tail fix. Source, object, library hashes and compile/link commands are recorded.
2. `python3 run-pool-replay.py` runs a CPU GLM5NEXT fixture through separate prefill, combined decode, changing active subsets, rollback, clear and refill. Old unified mode must abort with the recorded capacity assertion. New unified must match pool-cache-disabled logits; separated old/new must also match. Tensor callback counts full/incremental execution without changing arithmetic.
3. `python3 run-existing.py` runs existing test-llama-archs for glm5next and test-recurrent-state-rollback for its generated GGUF on CPU.
4. `python3 run-server.py` reuses the Stage2 request driver and exact request payloads with fixed current six-GPU placement. Runs: old unified DFlash, new unified DFlash, new unified pool-cache-disabled DFlash, new separated DFlash, new unified MTP. The initial MTP control also aborts on its separate unsupported reuse assertion. Each surviving server executes three serial, two groups of three concurrent, and three cache-reuse requests. Each output is bounded to64 tokens.

The real model is GLM-5.3-Flash-UD-Q4_K_XL, with DFlash2-Q8_0 for DFlash cases. Context8192, batch2048, ubatch64, three slots, F16 KV, layer split, fit off, explicit GPU UUIDs and tensor split. Environment LLAMA_GLM5_POOL_CACHE=0 supplies the independent full-recomputation control. Candidate binaries include no diagnostic model-arithmetic callback.

Full logs, binaries, float logits and memory samples stay in the workspace. The review archive contains scripts, summaries, manifests, source.patch, request/response JSON and hashes. No files are added under tests/.

After the MTP failure, `python3 build-final.py` builds the final guarded library without replacing V1. `python3 run-mtp-replay.py` reproduces the old MTP assertion and compares final unified multi-slot against forced recomputation, plus separated and single-slot old/new controls. `python3 run-final-pool.py` and `python3 run-final-existing.py` repeat the affected CPU checks on the final library. `python3 run-final-server.py` repeats only the failed MTP unified server case. The final source diff, binary hashes and server control are kept separate from V1.
