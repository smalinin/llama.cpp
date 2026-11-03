from pathlib import Path
import json,hashlib,shutil,subprocess
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');D=R/'docs/runtime-optimization/experiments/1017-hc-long-indexer-20261103';A=D/'artifacts';A.mkdir(parents=True,exist_ok=True)
mem=json.load(open(P/'memory-summary.json'));prof=json.load(open(P/'profile-results.json'));assert mem['complete']and prof['complete']
files=['config.json','metadata.cpp','run-metadata.py','metadata-results.json','metadata-driver.log','compile-metadata.log','placement.cpp','compile-placement.log','admit.py','gpu-map.json','payload-admission.json','run-payload.py','payload-driver.log','memory_monitor.py','summarize-memory.py','memory-summary.json','run-profile.py','ix-shim.cpp','inputs.json','profile-results.json','profile-driver.log','profile-startup-preflight-error.log','analyze-indexer.py','indexer-analysis.json','analysis.log','archive.py','validate.py','validation.json','validation.log','correlate-copies.py','copy-correlation.json','copy-correlation.log','topology-screen.json','design.md','write-report.py']
files += [p.name for p in P.glob('metadata-*.json')if p.name!='metadata-results.json']+[p.name for p in P.glob('metadata-*.log')if p.name!='metadata-driver.log']
files += ['payload-176128/'+n for n in ['results.json','server.log','placement.jsonl','inputs.json']]
files += ['sidecar/'+n for n in ['server.log','events.jsonl','placement.jsonl']]
files += [str(p.relative_to(P))for p in(P/'sidecar').glob('*-export.log')]
for n in sorted(set(files)):
 src=P/n;dst=A/n;assert src.is_file(),n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
provenance={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),'runtime_sha256':hashlib.file_digest((R/'ggml/src/ggml-backend.cpp').open('rb'),'sha256').hexdigest(),'libraries':prof['binary_sha256'],'probes':{n:hashlib.file_digest((P/n).open('rb'),'sha256').hexdigest()for n in ['metadata','placement.so','ix-shim.so']},'raw_profile_files':[{'locator':str(p),'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}for p in sorted((P/'sidecar').glob('*'))if p.suffix in ['.sqlite','.nsys-rep']]}
(A/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');(A/'RUNTIME.patch').write_bytes(subprocess.check_output(['git','diff','--','ggml/src/ggml-backend.cpp'],cwd=R))
(A/'README.md').write_text('''# Reproduction and evidence

Original preparation root: `/tmp/dsv41-hc-on-long-indexer-20261103`. Map absolute runtime/model/SDK/Nsight paths before replay. Do not execute historical runners in place or overwrite a loaded shared object. No model files or compiled probes are in this archive. Six library hashes, source delta, probe hashes, raw Nsight locators and hashes are in provenance.json; historical comparison inputs remain linked in the runner and prior evidence.

Compile metadata.cpp and placement.cpp against matching internal headers and libraries, then run no_alloc metadata and admit.py. Admit uses observed per-device overhead +64MiB guard, never just no_alloc projected free. run-payload.py performs a fresh actual176128 gate; keep its loader-only observer separate from the profiler. run-profile.py preserves the historical 100352 fixture and only profiles the diagnostic process with warm/pre/profile/post controls. It uses no eval callback or backend-interface changes and does not allocate probe GPU buffers or read tensor payloads. Selected exported CUDA op wrappers forward original args/streams once. The only probe CUDA synchronization is the labeled profiler drain after the HTTP response; it is outside prompt timing.

Build read-only diagnostic observer (from runtime root):

```bash
g++ -std=c++17 -O2 -fPIC -shared -DGGML_USE_CUDA -DGGML_USE_CPU -Iinclude -Isrc -Icommon -Iggml/include -Iggml/src -Ivendor -I/usr/local/cuda/include PROBE/ix-shim.cpp -Lbuild-aliceai/bin -Wl,-rpath,/home/sergei/Github/llama.cpp/build-aliceai/bin -Wl,--no-as-needed -lllama-common -Wl,--as-needed -lllama -lggml -lggml-base -L/usr/local/cuda/lib64 -lcudart -ldl -lpthread -o PROBE/ix-shim.so
```

The first profile preparation failed before GPU launch because an unrestricted git diff included non-UTF8 raw historical logs. The preserved preflight error is not a model failure. The final runner captures only the runtime source diff. Raw logs remain byte-for-byte, including terminal/progress whitespace. Profiler timing is diagnostic; clean prior OFF/ON/OFF remains the speed A/B.
''')
sha={str(p.relative_to(A)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()for p in sorted(A.rglob('*'))if p.is_file()and p.name!='SHA256.json'};(A/'SHA256.json').write_text(json.dumps(sha,indent=2)+'\n');print('archived',len(sha),'files',sum(p.stat().st_size for p in A.rglob('*')if p.is_file()))
subprocess.run(['git','add','-N','-f','--',str(D)],cwd=R,check=True)
tracked=set(subprocess.check_output(['git','ls-files',str(A)],cwd=R,text=True).splitlines());assert all(str((A/n).relative_to(R))in tracked for n in sha)
