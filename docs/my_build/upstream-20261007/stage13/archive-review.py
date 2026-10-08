#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'review-artifacts';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['capture-chain.cpp','target-matmul-controls.cpp','isolated-replay.cpp','run-chain-capture.py','run-target-matmul.py','run-isolated.py','prepare-isolated.py','analyze-chain.py','summarize-boundaries.py','analyze-target-matmul.py','analyze-isolated.py','analyze-profile.py','verify-integrity.py','archive-review.py','chain-capture-manifest.json','target-matmul-manifest.json','isolated-manifest.json','isolated-profile-manifest.json','chain-summary.json','boundary-summary.json','isolated-summary.json','kernel-summary.json','integrity.json','inputs/manifest.json']
for dataset in ['hc0','router0','post0','post2']:
 names.append(f'inputs/{dataset}/manifest.json')
 if dataset in ['hc0','router0']:names.append(f'inputs/{dataset}/config.txt')
for dirname in ['hc0-ada','router0-ada','post0-ada','post2-ada','hc0-ampere','router0-ampere','post0-ampere','post2-ampere']:
 names.append(dirname+'/results.jsonl')
for p in sorted((ROOT/'target-matmul-output').glob('*.json*')):names.append(str(p.relative_to(ROOT)))
for name in names:
 dest=OUT/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(ROOT/name,dest)
for source in ['chain-capture.log','target-matmul.log']:
 lines=(ROOT/source).read_text(errors='replace').splitlines()
 selected=[l for l in lines if l.startswith(('DONE ','REPLACED ','REPLACED_UPGATE '))]
 (OUT/(source.replace('.log','-completion.txt'))).write_text('\n'.join(selected)+'\n')
for p in sorted(ROOT.glob('*-profile.log')):
 selected=[l for l in p.read_text(errors='replace').splitlines() if l.startswith(('ggml_cuda_init:','  Device ','Generated:','\t/home/'))]
 (OUT/(p.stem+'.txt')).write_text('\n'.join(selected)+'\n')
(OUT/'README.md').write_text('''# Stage 13 diagnostic artifacts

HC/router localization on the Stage 12 FP32/scalar-attention/scalar-upgate control. Source HEAD fec769528cdb7e80f1f915efe031cb0d7cd61d36. No production source or installed server change.

Run the scripts in `/home/sergei/_my_sync/llama_upstream_review/stage13`, where previous stages and the immutable Stage 8 library snapshot remain available. Absolute paths in manifests refer to that workspace. `run-chain-capture.py` -> `analyze-chain.py` -> `summarize-boundaries.py` captures/localizes the first decode. `prepare-isolated.py` extracts two original GGUF weights and packs the captured HC operands; it also creates projection configs. `run-isolated.py` and `run-isolated.py --profile` run the isolated cases. `analyze-isolated.py` and `analyze-profile.py` produce their summaries. Export the four Nsight reports to SQLite before profile analysis. `run-target-matmul.py` -> `analyze-target-matmul.py` tests the model interventions. Output directories must not exist before running a harness. Run full-model and isolated GPU jobs sequentially.

Compile the two full-model harnesses with C++17/O2, repo include and ggml/include, linking llama, ggml and ggml-base from Stage 8 with the same absolute rpath. Compile isolated-replay.cpp with C++17/O2, ggml/include, linking ggml-cuda, ggml and ggml-base from that snapshot. CUDA Graphs are disabled by the run scripts; no production flags are introduced.

All 12 model replays passed; six reference SHA controls passed. HC+router preserves all 65 native argmax values at both widths, and width2/width4 full logits match each other exactly. Large scalar differences remain. Free-generation DSpark and real throughput are not tested for this candidate. All 88 isolated and 44 profiler cases are stable; all 56 profiler data files match their ordinary controls. All 12 Ada capture controls match; 11/12 Ampere controls match the Ada capture, with an architecture-dependent HC width4 exception documented in the report.

Tensor/logit dumps, weights, compiled binaries, full model logs, Nsight reports/SQLite, and raw-file-sha256.json stay in the local artifact directory. integrity.json records their checks and the hash of the raw index. Completion text files are extracted from full logs. artifact-sha256.json hashes this curated archive, excluding itself.
''')
hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'}
(OUT/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({'files':len(hashes)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
