#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'review-artifacts';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
names=['prepare-build.py','run-replay-control.py','run-free-generation.py','analyze-free.py','verify-integrity.py','archive-review.py','diagnostic-callback.h','server-experiment.patch','replay-control.cpp','build-manifest.json','replay-manifest.json','replay-control-summary.json','free-control-summary.json','free-summary.json','integrity.json','extended-replay.cpp','run-extended-replay.py','analyze-extended.py','extended-replay-manifest.json','extended-summary.json','n1-prior-comparison.json']
for name in names:shutil.copy2(ROOT/name,OUT/name)
for p in sorted((ROOT/'extended-replay-output').glob('*.json*')):
 d=OUT/'extended-replay-output';d.mkdir(exist_ok=True);shutil.copy2(p,d/p.name)
for config in ['snapshot-off','integration-off','candidate-off','candidate-n1','candidate-n3','snapshot-n3']:
 d=ROOT/'free-runs'/config;o=OUT/'free-runs'/config;o.mkdir(parents=True,exist_ok=True)
 for p in sorted(d.glob('*.json')):
  data=json.loads(p.read_text())
  if p.name.endswith('-response.json') and 'completion_probabilities' in data:
   data.pop('completion_probabilities');data['completion_probabilities_omitted_from_review_archive']=True
  (o/p.name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 for p in sorted(d.glob('*-prompt.txt')):shutil.copy2(p,o/p.name)
 if not config.startswith('snapshot-'):
  events=[l for l in (d/'server.log').read_text(errors='replace').splitlines() if l.startswith('DS14_EVENT ')]
  (o/'callback-events.txt').write_text('\n'.join(events)+'\n')
 selected=[l for l in (d/'server.log').read_text(errors='replace').splitlines() if 'print_timing:' in l or 'cleaning up before exit' in l]
 (o/'server-completion.txt').write_text('\n'.join(selected)+'\n')
lines=[l for l in (ROOT/'replay.log').read_text(errors='replace').splitlines() if l.startswith(('DONE ','REPLACED ','REPLACED_UPGATE '))]
(OUT/'replay-completion.txt').write_text('\n'.join(lines)+'\n')
shutil.copy2(ROOT/'build.log',OUT/'build.txt')
(OUT/'README.md').write_text('''# Stage 14 free-generation diagnostics

Source HEAD e235b05ba4163773de7c87da7fb3574ee61f7afb. Production source files and installed server are unchanged. The experimental directory copies Stage8/candidate-bin, replacing only libllama-server-impl.so. prepare-build.py generates a temporary server-context.cpp from the repository and applies server-experiment.patch, using the original build-glm53 flags and recorded object/static-library hashes. The rebuilt implementation uses diagnostic-callback.h, also included by replay-control.cpp.

Use the scripts in /home/sergei/_my_sync/llama_upstream_review/stage14 with the original model files, previous-stage artifacts and immutable Stage8 libraries available. Run prepare-build.py, then run-replay-control.py and verify its three SHA values against Stage13. Run run-free-generation.py for snapshot-off, integration-off, candidate-off, candidate-n1 and candidate-n3 sequentially. snapshot-n3 is a fresh original-library control on the baseline prompt with graphs disabled. After the free checks, run run-extended-replay.py and analyze-extended.py for the full95-token native history. Use analyze-free.py --controls-only before the speculative configurations; then analyze-free.py and verify-integrity.py. Output directories must not exist before harness runs. CUDA Graphs are disabled, one slot is used, and speculative verification widths are at most4.

There are four prompts and two requests per configuration (snapshot-n3 uses only baseline): one with n_probs5, one with n_probs0, each temperature0 and limit256. These are probability-output controls, not identical-parameter repeated benchmarks. Correctness compares complete token lists/content up to EOS or the limit. Callback events verify prefill/generation gating and replacement counts. Per-request timings include original operations, nested recomputation, allocations and synchronization. They are not a production speedup claim.

Raw full responses, server logs, replay logits, compiled binaries, the generated full server source and raw-file-sha256.json remain local. The curated responses omit completion_probabilities and explicitly mark that omission; token IDs, content, settings, timings and stop data are retained. Callback/completion text files are extracted from raw logs. integrity.json records the hash of the complete local raw index. artifact-sha256.json hashes this archive, excluding itself.
''')
hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'}
(OUT/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n');print(json.dumps({'files':len(hashes)+1,'bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}))
