# Stage 5 reproduction

Base: `cc7bfba3b7013e1c2287f7602854c9d093ef32bc`. Adapted upstream patches: `889edf43d` and `1b43d3116`.

The local QSA graph uses unscaled head scores, F32 finite visibility bias, separate KV streams and local MTP selection reuse. The adaptation replaces only the head-score materialization. Weights are one, the Lightning Indexer mask is derived by scaling the existing finite block bias by zero and casting to F16, and the original visibility bias is added in F32. This avoids one large uninitialized mask leaf per QSA layer. Key pooling is expanded before Q projection to retain the local scheduling order. The legacy graph is the default. `QWEN4EXP_FUSED_LID=1` explicitly selects the experimental path, and the existing `LLAMA_FUSED_LID_DISABLE=1` overrides it. The legacy graph also remains the path for indexer shapes other than 128 dimensions and four heads.

The tiled kernel retains F32 and BF16 key precision. Those types use a 32-key tile so static shared memory stays below 48 KiB; F16 and quantized types use the upstream 64-key tile. Both paths score eight tokens per block and use FP32 products. Smaller batches use the vector kernel. Existing 32/64-head dispatch is unchanged.

Artifacts:

- `source-version.json`, `source.patch` and `candidate-binary-sha256.json` describe the final source and binary. `after-bin` is the initial always-enabled experiment, `final-bin` adds the default gate, `review-bin` restores pooling order, and `candidate-bin` also removes the extra mask inputs. Versioned source patches and hashes retain their provenance. Binary snapshots, raw captures and logs stay local.
- `run-real.py before`, `after`, `final`, `review` and `candidate` run Qwen on fixed GPU UUIDs, at contexts 8192/65536, one/four slots and MTP off/on. The four-slot checks use concurrent requests with explicit slot IDs. All cases use the same layer split, F16 KV, batch 2048 and ubatch 512. Two full-prefill greedy repetitions are saved per prompt and slot, with prompt caching disabled. Long cases also use a prompt of 600 facts. Each process records loaded library paths and hashes, actual layer placement, model and compute buffers, and observed peak GPU memory.
- `qsa-check.cpp` compares the local score graph and fused op on the same backend against CPU, including padded views, multiple streams, incomplete key/token tiles, ReLU ties and large F32 values. Masked entries are excluded from NMSE, so a finite -1e9 mask cannot hide score errors. Tie selection order is checked separately. Its `reserve` mode measures graph allocator buffer sizes without computing large graphs.
- `capture-qsa.cpp` uses the public eval callback to save actual Qwen indexer queries, keys, final block scores and top-k, plus 32 steps of logits and greedy token IDs. Run with one/two streams. The callback affects scheduling and its timings are not performance evidence.
- `run-verification.py` waits for the complete baseline, serializes GPU verification and captures, then starts after-model checks. It stops on a failed command.
- `analyze.py` reads numerical, memory, capture and model evidence with the Python standard library. The optional `metrics.cpp` shared library accelerates float comparison; build it with `c++ -O3 -shared -fPIC metrics.cpp -o metrics.so`. It separates top-k order changes from changes in the selected set, aligns named tensors by step rather than callback order, excludes duplicate reshape/view callbacks, and checks loaded libraries. `run-final.py`, `run-review.py` and `run-candidate.py` preserve separate evidence for each local adjustment.

Build harnesses using the repository include directories and snapshot libraries. Supply `LD_LIBRARY_PATH` for each run; the copied server has an original build-directory RUNPATH. For another run choose a new output directory in the runners; existing case directories are deliberately not overwritten.

Model runs are sequential across configurations. Short and long warm performance estimates use the second repetition; four-slot values are medians across simultaneous slots and are not independent repetitions. These measurements establish practical behavior and memory use, not statistical significance of small timing differences.

AMD, Metal and Vulkan are not part of the installed CUDA build or hardware verification. Only the CUDA and existing backend-test parts of the upstream patches are transferred.

The isolated QSA score buffer shrinks substantially, but that does not establish a large reduction in the full model buffer. Default compatibility and experimental behavior are reported separately. The original cuBLAS path permits TF32; replaying captured Q/K with `NVIDIA_TF32_OVERRIDE=0` isolates the precision difference. Long-prompt experimental answers differ from the original path, so the experiment stays opt-in. Replay startup timings and rates for differing outputs are not proof of an application speedup.


Final whole-model result at context 65536: summed CUDA compute buffers change from 8707.07 to 8691.15 MiB for one slot (MTP off/on), and from 3437.44 to 3437.52 MiB for four slots. Experimental answers differ in 26/40 pairs; default answers and draft counters match in 60/60 pairs across all eight configurations. A large full-model memory saving or application speedup is not established. The last opt-in version was rerun at context 65536; the earlier experiment and default gate also cover context 8192.

`test-results.json` records the check versions and remaining activation criterion. `observed-memory-summary.json` computes the maximum simultaneous sum of six GPU samples, distinct from summing each GPU's maximum. `tf32-replay-comparison.json` excludes initialization timings. In the numerical summary, `ordinary_order_changes` counts changed top-k positions, while `ordinary_set_changes` counts rows with a changed selected set.

The Git copy contains helpers, patches, hashes, request/response records and aggregate results. Full raw captures and matching binary snapshots are required to rerun `analyze.py`; they are intentionally kept only in the local workspace. The runners refuse to overwrite existing server case directories. To repeat the suite, use a fresh artifact directory with matching before/after snapshots and retain the sibling baseline helper and Stage 4 GPU manifest.

Example harness build from this directory (paths are specific to this machine):

```bash
stage5_repo=/home/sergei/Github/llama.cpp
stage5_libs=$PWD/candidate-bin
c++ -std=c++17 -O2 -I "$stage5_repo/ggml/include" qsa-check.cpp -L "$stage5_libs" -Wl,-rpath,"$stage5_libs" -lggml-cuda -lggml-cpu -lggml-base -o qsa-check
c++ -std=c++17 -O2 -I "$stage5_repo/include" -I "$stage5_repo/ggml/include" capture-qsa.cpp -L "$stage5_libs" -Wl,-rpath,"$stage5_libs" -lllama -lggml-base -o capture-qsa
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH="$stage5_libs" ./qsa-check check
CUDA_VISIBLE_DEVICES=3 LD_LIBRARY_PATH="$stage5_libs" ./qsa-check check
CUDA_VISIBLE_DEVICES=4 LD_LIBRARY_PATH="$stage5_libs" ./qsa-check reserve
```
