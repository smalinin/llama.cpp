from pathlib import Path
import json,hashlib,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-kv-owner-audit-20261105';P=E/'artifacts';M=Path('/media/sergei/WW2T/_maya/project-maya-1.1.26/src/ds4/ds4_model.cu')
items=[(R/'src/models/deepseek41.cpp',[(1162,1176),(1236,1250),(1457,1480),(1505,1518),(1592,1607)]),(R/'src/llama-kv-cache-dsv4.cpp',[(1290,1320)]),(R/'src/llama-kv-cache.cpp',[(1398,1433)]),(R/'ggml/src/ggml.c',[(4033,4061)]),(R/'ggml/src/ggml-backend.cpp',[(1738,1753),(2135,2159),(2378,2389),(2610,2637),(2785,2792)]),(R/'ggml/src/ggml-cuda/ggml-cuda.cu',[(2718,2782)]),(M,[(9567,9599),(10137,10158),(11409,11431)])]
out=[]
for i,(f,ranges)in enumerate(items):
 s=f.read_text().splitlines();name=f'source-{i}.txt';(P/name).write_text(str(f)+'\n'+''.join('\n'.join(f'{k}: {s[k-1]}'for k in range(a,b+1))+'\n\n'for a,b in ranges));out.append({'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'ranges':ranges,'snapshot':name})
(P/'source-audit.json').write_text(json.dumps(out,indent=2)+'\n');shutil.copy2('/tmp/save-kv-owner-static.py',P/'save-static.py')
(E/'REPORT.md').write_text('''# KV owner and slot-lifetime audit, new host 2026-11-05

Status: read-only ownership capture running after residual commit cf1e5a461. Runtime source, libraries, defaults, model files and user presets unchanged. No new implementation or second commit.

The preceding queue4x/copies8/shared gate-up placement screens did not accelerate prefill. This stage records real tensor/source/view-root descriptors, layouts, flags, persistent vs compute/input buffers, buffer owner devices, and existing event record/wait/copy ordering. It deliberately avoids sched_get_tensor_backend on arbitrary tensors: that getter uses find_or_insert and can mutate the scheduler hash table. Public buffer-type device lookup is metadata-only.

The observer forwards original operations/copies/status/events; no extra GPU events, eval callbacks, CUDA API hooks or tensor-data reads. Selected target ubatches cover KV-only0/512/1536/2048, full-tail4096/4608/6144/7168/7676 and input-slot reuse. Full graph snapshots can perturb the armed request; pre/post unarmed controls are separate. No kernel/concurrency speed claim follows from metadata capture.

An adapted existing external scheduler fixture passed42 forwarding/failure/lifetime cases and graph-description root/source/owner coverage. Same frozen Q3/MXFP8 Engram/Q8 KV/DSparkK5/B8192/U512/P1/ctx100352/TopK2048/threads12/15360GPU bundles/default CPU lookup ON. Actual five-request8K capture and monitored512MiB floor are pending. See artifacts/build.json and run-controller.py.

## Static findings to validate against actual ownership

- Compressed-cache replicas are per source and actual reader device, not per every individual layer.
- At a source layer, prefill currently expands all remote replica SET_ROWS writes immediately. The existing LLAMA_DSV41_DEFER_CACHE_UPDATES default-on mechanism is gated by is_dsv41_decode, so it is inactive on512-token prefill.
- SET_ROWS is an in-place view of cache destination; its source order is values, indices, destination cache. A tensor named `(view)` can execute a real KV write and quantization.
- Prefill KV-only chunks return at layer20 with n_outputs0 and an empty pending-cache-update assertion. Dropping the decode-only gate without a KV-only flush would leave required remote updates pending and fail that assertion. Removing the assertion would permit missing replica state.
- Per-copy input arenas and existing cross-device events guard reuse. CUDA peer copies execute on the source stream, then the destination waits for the existing source copy event. These contracts must remain intact.
- Maya chunk workers publish staged new cache rows and readers pull them on their own stream, with explicit per-slot acknowledgements before stage reuse. This is a source-level architectural comparison, not a causal timing decomposition or permission to copy its numerics/transport formats.

Source hashes and narrow snapshots: artifacts/source-audit.json. The next design, if justified, should reuse deferred remote cache writes at the model-graph level and explicitly flush KV-only chunks, rather than removing scheduler fences.
''')
print('Static source audit and pending report saved.')
