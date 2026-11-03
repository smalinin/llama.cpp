# Reproduction

Runtime base828ee5b94, implementation6d038260e. Binary hashes in results.json. The build-info label still says309d6ca9b because it was generated before the implementation/default commits; use the recorded library hashes and active-branch evidence. No runtime source was changed during these runs.

Copy replay.py, replay-config.json, inputs.json, memory_monitor.py, shim.cpp and analysis scripts to an empty /tmp directory. Map runtime/model/Nsight/SDK paths explicitly; the archived original run.py keeps historical preparation paths. replay-config.json stores the frozen model argv/environment, not a universal model preset. Keep the runtime source clean, use free GPUs, and compile the observer against matching internal headers before launching replay. Do not overwrite a loaded .so.

Compile from the runtime root, replacing PROBE_DIR with the new /tmp directory:

```bash
g++ -std=c++17 -O2 -fPIC -shared -DGGML_USE_CUDA -DGGML_USE_CPU -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -I/usr/local/cuda/include PROBE_DIR/shim.cpp -Lbuild-aliceai/bin -Wl,-rpath,/home/sergei/Github/llama.cpp/build-aliceai/bin -Wl,--no-as-needed -lllama-common -Wl,--as-needed -lllama -lggml -lggml-base -L/usr/local/cuda/lib64 -lcudart -ldl -lpthread -o PROBE_DIR/shim.so
```

Run python3 replay.py in that empty probe directory. This launches native controls and a separate Nsight process; all requests are full uncached512/8192/98304 with16greedy outputs. The watcher enforces512MiB sampled free per GPU. NSYS/API permission is required; no sysctl is changed. Three captures are exported at shutdown. Analyze with analyze.py, summarize.py and gaps.py after completion, outside the clean measurements.

Optional scheduler follow-up: create an empty dispatch subdirectory, copy dispatch/replay.py as run.py plus memory_monitor.py and analyze.py there, retaining the parent replay-config.json/inputs.json/shim.so. Run that script to collect existing graph/scheduler/copy-stats diagnostics. It enables timing/stat counters, uses one cold request per prefix, and is not a new speed A/B. The analysis uses observed DS41/dflash node-count clusters to bucket wide target rebuilds; that classifier is not portable to another model.

No eval/abort callback is installed. The external observer wraps the current member decode and process_ubatch including its KV-only argument, and the member packed getter. It adds NVTX/logs; device drains occur only after the HTTP response and are labeled. The analysis subtracts packed feature_get children from inclusive bridge time. SQLite may contain non-UTF8 stdout strings; replacement decoding affects labels only, not numeric timing data.

trace-locators.json records large raw trace locations/hashes outside Git. The local summary/analysis/events, helper sources and manifests remain useful without those files. CPU/NVTX durations, kernel-duration sums, GPU unions and API-gap intersections are different metrics; do not add them as independent wall-time costs.
