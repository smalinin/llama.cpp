# Reproduction and evidence

Original preparation root: `/tmp/dsv41-hc-on-long-indexer-20261103`. Map absolute runtime/model/SDK/Nsight paths before replay. Do not execute historical runners in place or overwrite a loaded shared object. No model files or compiled probes are in this archive. Six library hashes, source delta, probe hashes, raw Nsight locators and hashes are in provenance.json; historical comparison inputs remain linked in the runner and prior evidence.

Compile metadata.cpp and placement.cpp against matching internal headers and libraries, then run no_alloc metadata and admit.py. Admit uses observed per-device overhead +64MiB guard, never just no_alloc projected free. run-payload.py performs a fresh actual176128 gate; keep its loader-only observer separate from the profiler. run-profile.py preserves the historical 100352 fixture and only profiles the diagnostic process with warm/pre/profile/post controls. It uses no eval callback or backend-interface changes and does not allocate probe GPU buffers or read tensor payloads. Selected exported CUDA op wrappers forward original args/streams once. The only probe CUDA synchronization is the labeled profiler drain after the HTTP response; it is outside prompt timing.

Build read-only diagnostic observer (from runtime root):

```bash
g++ -std=c++17 -O2 -fPIC -shared -DGGML_USE_CUDA -DGGML_USE_CPU -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -I/usr/local/cuda/include PROBE/ix-shim.cpp -Lbuild-aliceai/bin -Wl,-rpath,/home/sergei/Github/llama.cpp/build-aliceai/bin -Wl,--no-as-needed -lllama-common -Wl,--as-needed -lllama -lggml -lggml-base -L/usr/local/cuda/lib64 -lcudart -ldl -lpthread -o PROBE/ix-shim.so
```

The first profile preparation failed before GPU launch because an unrestricted git diff included non-UTF8 raw historical logs. The preserved preflight error is not a model failure. The final runner captures only the runtime source diff. Raw logs remain byte-for-byte, including terminal/progress whitespace. Profiler timing is diagnostic; clean prior OFF/ON/OFF remains the speed A/B.
