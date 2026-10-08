# Stage 4 reproduction

Base commit: `0a1ad5457`. CUDA patches: `1ab7e5ad2` and `9d3aba6b5`.

The server rebooted and one GPU was replaced during verification. Source diff and both binary snapshot hashes survived unchanged. Files from the interrupted run are in `interrupted-20261007/` and are excluded from the new comparison. The current hardware is recorded in `gpus.csv` and `gpu-topology.txt`; the paired real-model runner uses GPU UUIDs.

- `before-bin/`: libraries and tools at stage 3.
- `after-bin/`: libraries and tools with the two CUDA patches.
- `before-binary-sha256.json`, `after-binary-sha256.json`, `source-version.json`: provenance.
- `cuda-check.cpp`: finite deterministic GGML inputs, CPU reference, normalization fusion guards and small F16/BF16 matrix shapes. Graph timings include graph submission and synchronization overhead.
- `run-cuda-checks.py`: two GPU architectures, fused/unfused control and Nsight Systems kernel traces.
- `run-mat-tests.py`: non-contiguous matrix correctness and alternating before/after performance trials.
- `mat-perf-cases.txt`: eight small matrix and neighboring prefill cases for the existing backend tester.
- `run-real.py`: paired model checks on the same hardware, short and long Qwen/GLM5NEXT prompts, F16 KV, MTP/DSpark off/on. Dynamic library paths and actual layer placement are captured.
- `analyze-cuda.py`: numerical and kernel-count results.

Build the standalone numerical harness:

```sh
repo_dir=/home/sergei/Github/llama.cpp
review_dir=/home/sergei/_my_sync/llama_upstream_review/stage4
c++ -std=c++17 -O2 -I"$repo_dir/ggml/include" "$review_dir/cuda-check.cpp" -L"$review_dir/before-bin" -Wl,-rpath,"$review_dir/before-bin" -lggml-cuda -lggml-cpu -lggml-base -lggml -o "$review_dir/cuda-check"
python3 "$review_dir/run-cuda-checks.py"
python3 "$review_dir/run-mat-tests.py"
python3 "$review_dir/analyze-cuda.py"
```

Run GPU checks and real-model benchmarks sequentially. `run-real.py` creates case directories and refuses to overwrite existing cases; use a separate output directory for another full run.

An existing MMF stride assertion is reproduced by `mat-existing-stride-assert.txt` on both binaries. It is not included among passing cases. The first generic-operation trials produced NaNs for some padded views on CPU and CUDA; the cause was not established and these trials are excluded. `cuda-check.cpp` initializes the entire backing tensor with finite values for the valid non-contiguous tests.

The primary run completed 16 server launches and 112 completions. A further native GLM-DSA control used warmup plus six greedy requests per binary (14 completions). Run `validate-dsa-decode.py` after the other GPU jobs finish; it creates `validation-before/` and `validation-after/`. `analyze-dsa-validation.py` evaluates the last five warm requests.
