# Expert lane cost probe

This Linux/CUDA diagnostic measures quantized MoE FFN lanes. It does not change model placement or runtime policy. Source GGUFs and Maya files are read only. Build with CPU/CUDA enabled and `GGML_BACKEND_DL=OFF`:

```sh
cmake --build build-aliceai --target llama-moe-cache-calibrate -j8
python scripts/moe-cache-calibrate.py \
    --model /path/model-00001-of-00010.gguf \
    --cache /path/diagnostic-costs-v2.json \
    --gpus 2,3 --home-gpu 3 --layers 39 --threads 20 --repeats 5
```

`--gpus` and optional `--home-gpu` select physical nvidia-smi indices or full UUIDs. Home is the input/output GPU; destination is where pinned staging executes. Without `--home-gpu`, each measurement uses its destination as home. The probe verifies both PCI identities. Use idle devices. Host mask defaults to 6: up on GPU, gate/down on CPU (bits up=1, gate=2, down=4). The probe allocates full reference expert tensors on home, about 6 GiB for the Q3 Flash geometry, plus workspace.

Cases are `(tokens,picks)` = `(1,1)`, `(1,6)`, `(6,6)`, `(32,6)`, `(33,6)`, `(128,6)`, `(512,6)`. Modes are:

- `gpu_resident`: all three reference projections on home.
- `cpu_lane`: host-mask projections on CPU, other projections and final output on home; includes scheduler/activation transfers.
- `serial_h2d_gpu`: unique host expert slices copied synchronously from pageable mmap, followed by resident GPU execution; a separate diagnostic, without overlap.
- `pinned_staged`: the existing production `MUL_MAT_ID_STAGED` operator with four-expert waves, pinned double buffering, packing/H2D, actual quantized compute, and final output on home. Other host types keep the CPU fallback. Remote destinations include scheduler transfers to/from home. This mode does not model LRU hits.

Synthetic inputs use seed42, uniform `[-.25,.25]`, zero router logits and rotating distinct IDs. Input activations, logits and IDs are uploaded before **every** execution, outside the timer: the graph allocator can reuse their storage after their last reader. Version1 costs, produced without this reset, must not be reused. Finite outputs and CPU/staged errors relative to the GPU reference are reported. Serial/resident GPU error must be below0.01. CPU/GPU quantized rounding can differ; these operator errors are not a model-quality gate.

Optional `--fixtures /path/fixtures.json` replaces synthetic inputs with captured 512-token FFN activations, raw router logits and selected IDs. The manifest contains `model` and `rows`; each row has `case`, `name`, `shape`, `type`, `path`, `bytes`. Names are `ffn_norm-L`, `ffn_moe_logits-L`, `ffn_moe_topk-L`; dense binary types are F32/F32/I32, shapes `[5120,512,1,1]`, `[384,512,1,1]`, `[6,512,1,1]`. Capture noncontiguous views using their strides. Files are hashed into the calibration identity. Fixture cases measure widths33/128/512 using the first columns of each capture; repeated calls reset all inputs and use the same fixture. The serial diagnostic is omitted for fixtures. Capture callbacks must not run in timed measurements. Native `trace_nodes` is diagnostic only; the frontend rejects callback timings for cost publication.

Storage probes report warm mmap memcpy and verified aligned O_DIRECT reads when available. O_DIRECT bypasses the OS page cache, not necessarily the drive cache; these are not cold mmap-fault timings. Runtime staging packs on its calling thread; the probe does not introduce parallel file readers or change NUMA placement. Affinity, NUMA and CPU governor are recorded. Input uploads and output downloads are outside graph timing; all execution samples and min/median/max are retained after warmup2.

## Diagnostic persistence

The version2 identity includes probe/shared library/frontend hashes, tensor descriptors/source stats/sampled hashes, fixture hashes, home/destination UUID/PCI topology/driver, CPU affinity/NUMA/governor, and protocol. Sampled tensor hashes are not a full checkpoint checksum. Dynamic idle PCIe generation is excluded; negotiated width and maximum generation remain identity keys.

Reuse requires a width512 live recheck (warmup1, three trials). Drift beyond25% or20us refreshes the requested measurements. Identity changes or `--force` also refresh. Unknown versions, corrupt files, symlinks and concurrent writers are rejected. Publication uses an atomic same-filesystem rename. The previous cache survives probe failure, timeout or failed publication. Do not overwrite a version1 file; choose a new version2 path.

## Runtime startup auto

The runtime performs its own calibration when explicitly requested:

```sh
LLAMA_MOE_PREFILL_STAGE=auto ./llama-server \
    --model /path/model-00001-of-00010.gguf \
    --moe-expert-cache 64 ...
```

It stores `/path/model-00001-of-00010.gguf.moe-prefill-costs.json` beside the first GGUF shard. `LLAMA_MOE_PREFILL_STAGE_COSTS` optionally overrides this path. No Python, external diagnostic file, or manual cost values are required. The diagnostic version2 file above and the runtime startup file have different schemas and are not interchangeable.

Startup auto benchmarks the **loaded** expert projections through isolated CPU/staged FFN graphs: chunk512, the configured batch CPU threads, synthetic seed42/equal logits/rotating IDs, warmup2+three trials. It includes production pinned packing/H2D and return to the layer home. Calibration does not decode tokens, modify KV state, or populate the expert LRU. Candidates must have workspace/cache headroom plus1024MiB free reserve. A GPU must beat CPU by at least5% for every calibrated host layer in the group. Otherwise auto pins the host projections to the measured CPU backend. Stock scheduling can stream host weights to CUDA, so retaining the stock path is a separate choice. Resident projections stay on their GPU. Auto staging applies only to full512-token chunks; shorter prefill tails retain the CPU path, while the existing decode cache remains unchanged. Manual `=1` with optional `LLAMA_MOE_PREFILL_STAGE_DEVICE=CUDAx` keeps its existing width33..512 behavior and bypasses auto selection.

The adjacent file records loaded library hashes, model metadata/layout and sampled expert payloads, physical GPU/PCIe/driver identity, CPU/affinity/NUMA/governor, thread count, protocol and raw samples. A matching startup entry rechecks CPU and its best eligible staging candidate before reuse; only the rechecked candidate can be selected. Cost drift or changed identity triggers fresh calibration. Files are checksummed, locked and atomically replaced; foreign/corrupt files are preserved, with calibration retained in memory. An existing writer lock keeps the CPU fallback to avoid concurrent calibration. An unwritable model directory is reported explicitly; in-memory results can still be used. Metadata-fit does not benchmark or write files and reserves possible staging workspace on every active CUDA candidate. Final startup selection is frozen for that cache owner.

Startup calibration uses representative synthetic routes, a warm source and empty LRU. Real route locality, populated cache, cold storage and concurrent traffic can change the result; verify whole-model PP/TG and NLL on the intended workload. Layers without a host Q3_K projection and other architectures retain stock scheduling. Auto remains opt-in. It does not change GGUF types, placement, cache capacity settings or speculative policy. Changing CPU threads on a live context requires a new context to recalibrate.
