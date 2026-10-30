# Expert lane cost probe

This Linux/CUDA diagnostic measures expert FFN costs. It does not change model placement, cache capacity, or the runtime policy. Its JSON is a separate cost cache, not a route profile and not an input to `llama-server`.

Build with CPU/CUDA enabled and `GGML_BACKEND_DL=OFF`:

```sh
cmake --build build-aliceai --target llama-moe-cache-calibrate -j8
```

Run the frontend with Python 3.11+ and the dependencies of `gguf-py` (including NumPy):

```sh
python scripts/moe-cache-calibrate.py \
    --model /path/to/model-00001-of-00010.gguf \
    --cache /path/to/expert-costs.json \
    --probe build-aliceai/bin/llama-moe-cache-calibrate \
    --gpus 2,3 --layers 0,39 --threads 20 --repeats 5
```

`--gpus` selects physical `nvidia-smi` indices or full UUIDs. Each child sees one GPU, and its PCI bus ID is checked against that selection. Use idle devices. The default host mask is 6: gate/down on CPU and up on GPU. Bits are up=1, gate=2, down=4. Only DeepSeek41 target layers 0..39 are supported. The tool uses real quantized GGUF tensors but synthetic input activations and fixed expert IDs. It allocates the full three-projection tensor shapes on the selected GPU; about 6 GiB is needed for this Q3 geometry, plus workspace. Source GGUFs and Maya files are never written.

Cases are `(tokens, selected experts per token)` = `(1,1)`, `(1,6)`, `(6,6)`, `(32,6)`, `(512,6)`. Expert IDs rotate between iterations. The probe calls the existing MoE graph builder and checks CPU/GPU assignment for all three projections. It reports every timing sample and min/median/max after two warmup calls:

- `gpu_resident`: all three projections already on the same GPU.
- `cpu_lane`: host-mask projections forced onto CPU; the other projections and input/output are on GPU. Includes activation transfers and scheduler/graph execution.
- `serial_h2d_gpu`: the unique host-mask expert rows are copied from pageable mmap, then the graph runs entirely on GPU. Includes synchronous weight transfers with no overlap. This is not the pinned asynchronous cache worker or staging implementation.

Input transfers and output downloads are outside the timed interval. Backend CUDA graph capture/reuse follows the current library defaults. The same input and expert IDs are used for all three modes. Finite outputs are required; max absolute and relative L2 error against resident GPU are saved. CPU/GPU rounding can differ. The synthetic equal-weight routing can activate rare experts on inputs they do not receive in normal generation. These comparisons are diagnostics, not an NLL/PPL quality gate. Resident/recopied GPU results must agree within 0.01 maximum absolute error.

Storage probes report warm mmap-to-RAM `memcpy`, and aligned `O_DIRECT` reads when supported. Direct reads are checked against mapped source bytes. Allocation/open time and the verification checksum are outside the timer. `O_DIRECT` bypasses the OS page cache, not necessarily the drive's cache. These are not cold mmap-fault timings. The tool does not drop the system page cache.

## Persistence

The cache key contains the probe/shared llama and ggml library hashes, frontend hash, tensor names/types/shapes/strides/offsets, source paths/stat identity and sampled hashes, GPU UUID/PCI topology/driver, CPU/affinity/NUMA/governor, and benchmark protocol. Weight checks sample the first/last 128 bytes of each tensor; they are not a full checkpoint checksum. Idle PCIe link generation is excluded, since power saving changes it. Current negotiated width and maximum generation are included.

A matching cache runs a short recheck on cases `(1,1)` and `(1,6)` before reuse (one warmup, three trials). A median must stay within 25% or 20 us, whichever is larger. Drift refreshes the full cache. Identity changes also refresh it. `--force` refreshes explicitly. The previous cache survives a failed probe, timeout, or atomic replacement failure. A directory lock rejects another writer; a forced kill can leave a stale `.lock` directory. Remove it only after checking that no writer remains. Unknown/corrupt files are rejected and preserved, even with `--force`.

The cache is published by a same-filesystem atomic rename after all measurements complete. GPU/CPU load, clocks, page residency, and thermal state still affect costs; a matching identity does not guarantee stable performance. The recheck is a diagnostic drift check, not approval to change execution policy.

Peer-to-peer transfers, pinned/overlapped streaming, cold page faults, real route samples, nonuniform capacities, and adaptive dispatch remain separate work. Do not derive whole-model PP/TG or a production CPU/streaming decision from these FFN microbenchmarks alone.
