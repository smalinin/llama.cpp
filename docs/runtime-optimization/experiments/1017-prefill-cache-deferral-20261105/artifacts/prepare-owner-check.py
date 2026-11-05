from pathlib import Path
import shutil,json
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';O=R/'docs/runtime-optimization/experiments/1017-kv-owner-audit-20261105/artifacts';D=P/'owner-on';D.mkdir();shutil.copy2(O/'probe.cpp',D/'probe.cpp');shutil.copy2(O/'probe.so',D/'probe.so');shutil.copy2(P/'memory_monitor.py',D/'memory_monitor.py')
s=(O/'run.py').read_text().replace('/1017-kv-owner-audit-20261105/artifacts','/1017-prefill-cache-deferral-20261105/artifacts/owner-on').replace('19101','19103').replace("='prefill-kv-owner'","='prefill-deferred-owner'")
s=s.replace("cmd=cfg['command'][:];", "cmd=cfg['command'][:];cmd[cmd.index('--ctx-size')+1]='92160';")
s=s.replace("LD_LIBRARY_PATH=str(R/'build-aliceai/bin')", "LD_LIBRARY_PATH=str(P.parent/'candidate')+':'+str(R/'build-aliceai/bin')")
s=s.replace("env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None);", "env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None);env['LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES']='1';")
s=s.replace("j['ready_seconds']=time.monotonic()-t;persist();", "j['ready_seconds']=time.monotonic()-t;j['llama_maps']=[x for x in Path(f'/proc/{proc.pid}/maps').read_text().splitlines()if'/libllama.so'in x];assert j['llama_maps']and all(str(P.parent/'candidate')in x for x in j['llama_maps']);persist();")
s=s.replace("j={'complete':False,", "assert json.loads((P.parent/'precision-results.json').read_text())['complete'];j={'complete':False,")
s=s.replace("'kind':'", "'kind':'Actual ON deferral graph ownership and KV-only flush confirmation, same92160context and private library; not clean speed timing. ")
(D/'run.py').write_text(s);compile(s,str(D/'run.py'),'exec');shutil.copy2('/tmp/prepare-deferral-owner-check.py',P/'prepare-owner-check.py');print('Actual ON owner check prepared; precision gate required.')
