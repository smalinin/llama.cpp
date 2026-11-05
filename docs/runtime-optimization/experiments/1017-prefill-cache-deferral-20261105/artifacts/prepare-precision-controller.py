from pathlib import Path
import json,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';O=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts'
s=(O/'precision.py').read_text().replace('/1017-cpu-lookup-overlap-20261105/artifacts','/1017-prefill-cache-deferral-20261105/artifacts');s=s.replace(" while not(P/'precision-release').exists():time.sleep(.2)"," assert json.loads((P/'final-context.json').read_text())['complete'];assert json.loads((P/'graph-checks.json').read_text())['complete']")
s=s.replace("native=json.loads((P/'native-results.json').read_text());assert native['complete'] and native['all_outputs_counters_exact']","build=json.loads((P/'build.json').read_text());cfg['source_library_hashes']={f:sha(f)for f in cfg['source_library_hashes']};cfg['command'][cfg['command'].index('--ctx-size')+1]='92160'")
s=s.replace('candidate/libggml-base.so.0','candidate/libllama.so.0').replace('Two98K','Two88K').replace('Fresh new-host OFF vs ON only','Approved prefill cache deferral, same private llama/ctx92160, OFF vs ON only')
s=s.replace("env.update(GGML_SCHED_CPU_LOOKUP_EARLY=flag,", "env.update(LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=flag,")
s=s.replace("j={'complete':False,", "j={'complete':False,'build':build,'config':cfg,")
s=s.replace("assert sha(P/'candidate/libllama.so.0')==j['candidate_sha256']", "assert sha(P/'candidate/libllama.so.0')==j['candidate_sha256'];assert all(sha(R/x['source'])==x['source_sha256']for x in build['compile'])")
s=s.replace("if not ready and '\\nREADY\\n'in text:ready=True;mon.inference();print('READY_PRECISION',flag,flush=True)", "if not ready and '\\nREADY\\n'in text:\n      arm['loaded_llama_maps']=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert arm['loaded_llama_maps']and all(str(P/'candidate')in x for x in arm['loaded_llama_maps']);ready=True;mon.inference();persist();print('READY_PRECISION',flag,flush=True)")
s=s.replace("assert flag=='0'or any(re.search('early_transitions=[1-9]',x)for x in arm['activation']);", "")
(P/'precision.py').write_text(s);compile(s,str(P/'precision.py'),'exec');shutil.copy2('/tmp/run-deferral-precision.py',P/'prepare-precision-controller.py')
