from pathlib import Path
import json,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';O=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts'
for name in ['precision-harness','precision.cpp','precision-inputs.json','fault.so','fault.cpp']:shutil.copy2(O/name,P/name)
s=(O/'precision.py').read_text().replace('1017-prefill-cache-deferral-20261105','1017-verify-moe5plus1-20261105');s=s.replace("assert json.loads((P/'final-context.json').read_text())['complete'];assert json.loads((P/'graph-checks.json').read_text())['complete']","assert json.loads((P/'admission.json').read_text())['admit'];assert json.loads((P/'operator-run.json').read_text())['complete']")
s=s.replace("cfg['source_library_hashes']={f:sha(f)for f in cfg['source_library_hashes']};","cfg['source_library_hashes']={f:sha(f)for f in cfg['source_library_hashes']};cfg['source_files']={f:sha(R/f)for f in cfg['source_files']};")
s=s.replace("ref=None\n for flag in ['0','1']:","reference_path=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts/precision-0/results.json';reference=json.loads(reference_path.read_text());assert reference['complete'];j['reference_results']={'path':str(reference_path),'sha256':sha(reference_path),'context':92160,'input_sha256':sha(P/'precision-inputs.json'),'scope':'Same-day frozen exact OFF reference on unchanged hardware/placement/cache/kernel libraries; current candidate changes only bounded5+1MoE selection for sixqueries.'};ref={(r['corpus'],r['key']):r['result']for r in reference['cases']};persist()\n for flag in ['1']:")
s=s.replace("env.update(LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=flag,","env.update(LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES='0',LLAMA_DSV41_VERIFY_MOE_WIDE=flag,")
s=s.replace("j={'complete':False,'build':build", "j={'complete':False,'build':build")
s=s.replace('Approved prefill cache deferral, same private llama/ctx92160, OFF vs ON only, not historical arithmetic/state.','Bounded MoE5+1 at sixqueries, private candidate/ctx92160, explicit wide1/prefilldeferral0. Compare to frozen same-day exact OFF reference; no unrelated arithmetic/state change.')
s=s.replace(' and4total seed-failure/retry checks',' and2candidate seed-failure/retry checks')
s=s.replace("'candidate_sha256':sha(P/'candidate/libllama.so.0')","'candidate_sha256':sha(P/'candidate/libllama.so.0')")
# Compare each completed result group as it is written, preserving a numerical failure early.
needle="     assert time.monotonic()-start<3600;time.sleep(1)";replacement="""     progress_file=d/'results.json'
     if progress_file.exists():
      try: partial=json.loads(progress_file.read_text())
      except json.JSONDecodeError: partial=None
      if partial:
       for row in partial.get('cases',[]):
        key=(row['corpus'],row['key'])
        if row['result']!=ref[key]:
         arm['early_mismatch']={'corpus':key[0],'key':key[1]};persist();raise AssertionError(('candidate precision mismatch',key))
     assert time.monotonic()-start<3600;time.sleep(1)""";assert needle in s;s=s.replace(needle,replacement)
s=s.replace("shutil.copy2('/tmp/run-deferral-precision.py',P/'run-precision.py')",'')
(P/'precision.py').write_text(s);shutil.copy2('/tmp/prepare-moe5plus1-precision.py',P/'prepare-precision.py')
print('PRECISION_CONTROLLER_READY')
