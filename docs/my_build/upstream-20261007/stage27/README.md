# Stage27: DeepSeek-V4.1 tail rollback

User-authorized narrow source fix after the Stage0-26 testing cycle was closed. Stop for review before commit. No installed server or profile replacement.

The only production change is `src/llama-kv-cache.cpp`. Removing a suffix of a DeepSeek-V4.1 raw SWA sequence with an independent stream now places the allocation head at the freed cell with the lowest logical position. Previously the physical-index scan could select cell 0 instead of 766 when verification crossed the 768-cell ring boundary. Full clear, bounded removal, shared streams, other architectures and compressed KV keep their old policy. This restores scalar placement after tail rollback; it does not change CUDA arithmetic or claim that the old allocation was invalid memory access.

`source.patch` records the exact production diff. The allocator harness uses real find_slot/apply_ubatch/seq_rm without model weights: 27 cases, seven old-library failures, zero candidate failures. Cases cover ring crossings, retained prefix cells, independent sequence streams and scope guards.

Model replay reuses the Stage26 histories and actual schedules. Only the assertion requiring the OLD physical layout is removed (see `replay.patch`); raw-plan pre/post consistency checks stay enabled, and the analyzer requires the NEW traces to match scalar layout. Actual draft tokens, decode widths, positions and rollback calls are unchanged. The cached actual replay warms the full recorded fresh schedule before partial restore. All Stage24 diagnostic arithmetic and raw-layout headers are byte-identical.

Six model replays cover baseline95 widths1/4, fresh/cache scalar1024 controls and the two recorded rollback prefixes. Full logits on the 405/786 correct-prefix rows are compared in-process with scalar; focal vectors are also retained. Four complete control matrices have unchanged SHA256. The actual schedule continues beyond its correct prefix, but numeric equality is asserted only where its history is still native. All physical trace rows are compared.

The server check uses the unchanged Stage26 diagnostic capture server plus the fixed libllama. It runs exactly two long N3 requests (fresh/cache), each 1024 tokens with ignore_eos, and compares tokens/content/stop fields with the Stage25 native responses. This validates those failures under existing diagnostic arithmetic. It does not validate general production N3 bit equivalence, production throughput, native fresh/cache equivalence or session restore.

## Reproduction

Paths in scripts refer to this machine. Existing output directories are immutable; use a separate copy for a rerun. Reuse Stage8 preserved libraries, the original build-glm53 object files, Stage24 callback headers, Stage25 native responses and Stage26 inputs/schedules. Model/draft files and six-GPU order are recorded in manifests. Run GPU processes sequentially.

1. Apply/review `source.patch` on source HEAD fdc24e5ef; `python3 build-candidate.py` rebuilds only the KV object and relinks libllama into candidate-bin. It checks the original libraries and records reused object hashes.
2. `python3 run-tail-control.py` builds and executes the allocator regression against old/candidate libraries.
3. `python3 prepare-replay.py` prepares/builds the target-only driver. `python3 run-replay.py --cases cases.txt --output model-output --prefix replay` executes six cases; `python3 analyze-replay.py` writes replay-summary.json.
4. After replay finishes, `python3 prepare-server.py` combines the Stage26 diagnostic server with fixed libllama. `python3 run-server.py` runs the two requests; `python3 analyze-server.py` writes server-summary.json.
5. Record GPU state, then `python3 verify-integrity.py` checks build/source/input/runtime-library hashes and all result criteria. `python3 archive-review.py` writes the compact review archive only after integrity succeeds.

Build flags, commands, loaded library mappings, input hashes and exit codes are in manifests. Binaries, full logits, checkpoints and full logs stay in the workspace; raw-file-sha256.json records their sizes and hashes. The Git archive keeps scripts, manifests, request/response JSON, row metadata, traces and selected events. Prior stages are not rewritten.
