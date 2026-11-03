from pathlib import Path
import json,hashlib,shutil,subprocess,difflib
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');D=R/'docs/runtime-optimization/experiments/1017-candidate-pool-local-20261103';A=D/'artifacts';s=json.load(open(P/'summary.json'));assert s['complete'];A.mkdir(parents=True,exist_ok=True)
for src in sorted(P.iterdir()):
 if not src.is_file()or src.suffix not in ['.py','.cpp','.log','.json','.jsonl','.patch']or src.name=='SHA256.json':continue
 shutil.copyfile(src,A/src.name)
for folder in ['long-native','fresh-on-native','bench-off-pre','bench-on','bench-off-post','payload-176128']:
 for src in sorted((P/folder).iterdir()):
  if not src.is_file()or src.suffix not in ['.json','.jsonl','.log']:continue
  dst=A/folder/src.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
shutil.copyfile('/tmp/dsv41-m2-residual-profile-20261103/inputs.json',A/'bench-inputs.json')
shutil.copyfile('/tmp/dsv41-hc-on-long-indexer-20261103/metadata.cpp',A/'metadata.cpp')
shutil.copyfile(R/'src/models/deepseek41.cpp',A/'deepseek41.final.cpp')
(A/'FINAL_GUARD.patch').write_text(''.join(difflib.unified_diff((P/'llama-context.measured.cpp').read_text().splitlines(True),(P/'llama-context.final.cpp').read_text().splitlines(True),fromfile='llama-context.measured.cpp',tofile='llama-context.final.cpp')))
probes=['metadata-trace.so','trace.so','long','fresh-on','operator','operator-final','transfer-audit.so','placement.so']
prov={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),'precision_libraries':json.load(open(P/'precision-config.json'))['source_library_hashes'],'final_libraries':s['libraries'],'final_source_sha256':s['source_sha256'],'probes':{n:hashlib.file_digest((P/n).open('rb'),'sha256').hexdigest()for n in probes},'metadata_helper':{'locator':'/tmp/dsv41-hc-on-long-indexer-20261103/metadata','sha256':hashlib.file_digest(open('/tmp/dsv41-hc-on-long-indexer-20261103/metadata','rb'),'sha256').hexdigest()},'metadata_helper_note':'Reused helper compiled against unchanged internal headers/common fit implementation; runtime shared libraries are final for each metadata phase.','compiled_probe_binaries_archived':False};(A/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
(A/'README.md').write_text('''# Reproduction and scope

Original root `/tmp/dsv41-candidate-pool-20261103`; runtime `/home/sergei/Github/llama.cpp`, branch my_maya. Map all runtime/model/SDK paths in an empty directory before replay. Do not execute these historical orchestration scripts in the archive, overwrite a loaded .so, or replay the mode-switch fixture as a production server configuration. Set the opt-in before loading a new context. Model files/binaries are not copied here. Runtime/probe hashes and phase identities are in provenance.json.

Precision phase used MEASURED.patch: full mutable OFF/ON/OFF36 groups plus a separate fresh constantON12 groups, exact target/draft logits/state/taps/NLL/rollback, two seed faults per phase. The mutable fixture retains an ON-grownCUDA2 buffer at shutdown and emits one recorded size warning. ConstantON from constructor reserves enough and exits0 without that warning. Do not suppress a warning in a normal server because this fixture has a separate, explained one.

run-fresh-on.measured.py accidentally read the old36-group results file while expecting12 after its native child already exited0. The raw manifest remains complete:false/error. check-fresh-on.py independently validates the actual command/output file, all12 groups against the OFF reference, failure/retry, floor and clean lifecycle. Corrected run-fresh-on.py is retained for future replay, not claimed as the measured runner.

FINAL_GUARD.patch only requires the next-source search result to be below n_layer. It rejects an exhausted24-layer search; the tested model has40layers/next24. Final metadata budgets/routes/owners were repeated; final native OFF/ON/OFF, actual176128 and pointer-copy proof use the final libraries. Precision results retain their earlier hashes. No GPU arithmetic, dtype or default changed in this tightening.

The first operator.cpp/operator-results.json/operator.log is rejected as two-GPU evidence: implicit graph_compute allocation reset backend hints and ran both routes onCUDA0. operator-final.cpp explicitly allocates after assigning, checks actual backend IDs, compares80 boundary/nonfinite cases and32 changing inputs on isolated fresh CUDA backends. Generic shape-changing graph update fallback is separate from isolated fixed32 capture/replay stats. The corrected operator is run after the native/memory series so its build/run cannot perturb benchmark timing.

Clean bench.py: three separate OFF/ON/OFF processes,27 full uncached requests with256greedy outputs. No LD_PRELOAD/profiler/callback/calibration/build in timed runs. Warmup points are excluded from medians.98K has one measured request per arm; retain drift/range limitations.

Actual176128/pointer audit is a separate process, not a speed benchmark. transfer-audit.cpp reads CPU tensor metadata, intercepts full-size CUDA copy API args once, creates no GPU buffers and adds no CUDA sync/payload reads. analyze-audit.py requires both exact pointers, full bytes/shape/strides/type, protected destination clone, actual consumer and expected CUDA ordinals; unrelated/partial matches do not count. API payload is not a hardware DMA counter.

Raw logs keep their original bytes/whitespace. Compiled probe locators/hashes and all sources are preserved; compilation uses the matching runtime internal headers and linked libraries as recorded in scripts/logs. Original late postprocessing errors remain visible alongside corrected validators; acceptance follows summary.json and the main report, not a raw complete flag alone.
''')
ss={str(p.relative_to(A)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest()for p in sorted(A.rglob('*'))if p.is_file()and p.name!='SHA256.json'};(A/'SHA256.json').write_text(json.dumps(ss,indent=2)+'\n');subprocess.run(['git','add','-N','-f','--',str(D)],cwd=R,check=True);tracked=set(subprocess.check_output(['git','ls-files'],cwd=R,text=True).splitlines());assert all(str((A/n).relative_to(R))in tracked for n in ss);print('ARCHIVE_DONE',len(ss),sum(p.stat().st_size for p in A.rglob('*')if p.is_file()))
