from pathlib import Path
import hashlib,json,subprocess,ast
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent
w=json.loads((P/'width-results.json').read_text());s=json.loads((P/'width-summary.json').read_text());b=json.loads((P/'build.json').read_text())
assert w['complete'] and w['all_outputs_state_exact'] and s['complete']
sha=lambda f:hashlib.file_digest(Path(f).open('rb'),'sha256').hexdigest()
for f,h in w['config']['source_library_hashes'].items():assert sha(f)==h,f
for f,h in w['source_hashes'].items():assert sha(R/f)==h,f
assert sha(P/'candidate/libllama.so.0')==b['private_llama_sha256']
assert sha(P/'reference-llama-graph.cpp')==hashlib.sha256(subprocess.check_output(['git','show','HEAD:src/llama-graph.cpp'],cwd=R)).hexdigest()
assert not subprocess.check_output(['git','diff','--','src/models/deepseek41.cpp','ggml','common','include','src/llama-graph.h'],cwd=R,text=True)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
for f in P.glob('*.py'):ast.parse(f.read_text())
rows='\n'.join(f"| {r['prefix']} | {r['queries']} | {r['off']['median_ms']:.3f} | {r['on']['median_ms']:.3f} | {r['latency_delta_pct']:+.2f}% |"for r in s['rows'])
p=P.parent/'REPORT.md';text=p.read_text();text=text.replace('Status: primitive exactness PASS, private candidate built, fresh metadata admission PASS. Paired full-model width controls are running; no full-model correctness/native speed/default promotion claim yet.','Status: primitive exactness and the paired fixed-prefix width screen PASS. No complete lifecycle/two-corpus/native speed/default promotion claim yet.')
text+='''\n\n## Completed paired target-width screen

Same private module in OFF/ON, only FLOAT_WIDE0/1 changes; MoE_WIDE1 in both. Both fresh processes use the newly rebuilt CUDA/base modules. Each prefix has five measured repetitions per width per forward/reverse pass, plus retained warmups. Target decode+sync only; snapshot restore, bridge/draft, raw output transport and hashing are outside the timer. No callback/CUPTI/interposer. All observations remain, including cold/variable rows. This is a raw-logit target screen, not server GPU-sampler TG.

| Prefix tokens | Target queries | OFF median ms | ON median ms | Latency change |
| --- | --- | --- | --- | --- |
'''+rows+f'''

All156 raw observations per arm and24 sampled target-state checkpoints per arm match across flags, including base state/proposal IDs and first-row versus width1. Both processes exit0 with no floor breach, forced kill or CUDA/OOM/compute-buffer expectation warnings. Actual sampled minima: {', '.join(f'{k} {v:.3f}MiB'for k,v in s['actual_floor_mib'].items())}. All measured frontend graphs are reused; no fresh CUDA-instantiation counter was added to this timing. [Raw manifest](artifacts/width-results.json), [OFF rows](artifacts/width-off/results.json), [ON rows](artifacts/width-on/results.json), [summary](artifacts/width-summary.json).

The phase is still opt-in/OFF. This screen exercises both widths and state restores, but it does not replace the full rejected-window rollback/taps/two-corpus/fault matrix, native PP/TG, or the MoE0-only cohort. A lower target-call median is not an end-to-end throughput claim. Next: confirm the actual floating-node/dispatch reduction separately, then full lifecycle and native controls. No Phase C shared-weight selector change or new commit is included.
'''
p.write_text(text)
out=dict(complete=True,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),primitive_evaluations=360,raw_observations_per_arm=156,state_checkpoints_per_arm=24,native_speed_gate=False,full_lifecycle_gate=False,default_on=False,production_modules_unchanged_by_phase_b=True,
artifact_hashes={f:sha(P/f)for f in ['operator-results.json','operator-run.json','build.json','admission.json','width-results.json','width-summary.json','runtime.patch']})
(P/'final-checks.json').write_text(json.dumps(out,indent=2)+'\n')
print([(r['prefix'],r['off']['median_ms'],r['on']['median_ms'],r['latency_delta_pct'])for r in s['rows']if r['queries']==6])
