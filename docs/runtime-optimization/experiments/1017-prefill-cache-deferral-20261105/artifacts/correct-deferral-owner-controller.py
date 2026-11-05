from pathlib import Path
import json,shutil,hashlib
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';D=P/'owner-on';B=P/'owner-default';assert not B.exists();r=json.loads((D/'counts-results.json').read_text());assert r['complete']and r['environment'].get('LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES')is None
original_hash=hashlib.sha256((D/'counts-results.json').read_bytes()).hexdigest();D.rename(B);D.mkdir()
for name in ['probe.cpp','probe.so','memory_monitor.py','analyze.py']:
 shutil.copy2(B/name,D/name)
s=(B/'run.py').read_text();needle=";env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None)";assert s.count(needle)==1;s=s.replace(needle,needle+";env['LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES']='1'")
s=s.replace("j={'complete':False,", "assert env['LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES']=='1';j={'complete':False,")
(D/'run.py').write_text(s);compile(s,str(D/'run.py'),'exec')
# Relabel via a companion record; leave the original raw manifest/log/scopes intact.
(P/'owner-default-recovery.json').write_text(json.dumps({'complete':True,'original_manifest_sha256':original_hash,'actual_mode':'Default OFF, env flag absent; private library and92160context correct.','reason':'Generator searched env.pop followed by semicolon, but source line ended after the closing parenthesis. It failed to insert LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=1. Initial source kind string was misleading; actual environment and eager graph prove OFF.','baseline_path':'owner-default','precision_effect':'None: precision-results.json records explicit0and1; both full-model cohorts genuinely ran their requested mode. No runtime source/binary changed for this correction.','new_controller':'owner-on/run.py now explicitly inserts and asserts flag1 before launching.','native_started':False},indent=2)+'\n')
shutil.copy2('/tmp/correct-deferral-owner-controller.py',P/'correct-owner-controller.py');shutil.copy2('/tmp/run-deferral-native-after-owner.py',P/'initial-native-after-owner.py')
st=json.loads((P/'STATE.json').read_text());st.update(stage='Precision OFF/ON fully exact. Ownership diagnostic initially lacked flag and is retained as same-context default baseline. Corrected actual ON trace running; native not started.',sessions={'owner_on':'current corrected exec session','precision':'4097 completed'});(P/'STATE.json').write_text(json.dumps(st,indent=2)+'\n');print('Initial trace preserved as actual default baseline; corrected ON controller asserts its flag.')
