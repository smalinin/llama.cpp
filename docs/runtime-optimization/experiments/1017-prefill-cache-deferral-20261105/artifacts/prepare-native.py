from pathlib import Path
import shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';O=R/'docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts'
s=(O/'native.py').read_text().replace('/1017-cpu-lookup-overlap-20261105/artifacts','/1017-prefill-cache-deferral-20261105/artifacts').replace('candidate/libggml-base.so.0','candidate/libllama.so.0')
s=s.replace("check=json.loads((P/'scheduler-results.json').read_text());assert check['complete'] and len(check['cases'])==36", "check=json.loads((P/'precision-results.json').read_text());assert check['complete'] and check['all_groups_exact'];assert json.loads((P/'final-context.json').read_text())['complete'];cfg['command'][cfg['command'].index('--ctx-size')+1]='92160'")
s=s.replace("sha(R/'ggml/src/ggml-backend.cpp')", "sha(R/'src/models/deepseek41.cpp')")
s=s.replace('128greedy output tokens,127 TG intervals','256greedy output tokens,255 TG intervals').replace('ctx100352','ctx92160').replace('512/8K warm+two measured,98K warm+one measured','512/8K/88K warm+two measured each').replace('Q3/Q8KV/K5/pmin.5/full/B8192/U512/P1/TopK2048','Q3/Q8KV/K5/pmin.5/full/B8192/U512/P1/TopK2048')
s=s.replace("plan=[(n,k)for n in [512,8192,98304]for k in(['warmup','measured','measured']if n!=98304 else['warmup','measured'])]", "plan=[(n,k)for n in [512,8192,90112]for k in ['warmup','measured','measured']]")
s=s.replace('GGML_SCHED_CPU_LOOKUP_EARLY=flag','LLAMA_DSV41_PREFILL_DEFER_CACHE_UPDATES=flag').replace("='19093'","='19104'").replace('127.0.0.1:19093','127.0.0.1:19104').replace("='lookup-'+label","='cache-defer-'+label")
s=s.replace("if'libggml-base'in x", "if'/libllama.so'in x")
s=s.replace("    while not(P/'native-release').exists():assert proc.poll()is None;time.sleep(.2)\n",'')
s=s.replace("request(ids[:n],128)","request(ids[:n],256)").replace("len(v['tokens'])==128","len(v['tokens'])==256").replace("'tg':127000/t['predicted_ms']","'tg':255000/t['predicted_ms']")
s=s.replace(";assert flag=='0'or any(re.search('early_transitions=[1-9]',x)for x in a['activation']),('candidate inactive',a['activation'])",'')
s=s.replace("cmd=cfg['command'][:];cmd[cmd.index('--port')", "env.pop('GGML_SCHED_CPU_LOOKUP_EARLY',None);cmd=cfg['command'][:];cmd[cmd.index('--port')")
s=s.replace("'source_sha256':sha(R/'src/models/deepseek41.cpp'),", "'source_sha256':sha(R/'src/models/deepseek41.cpp'),'source_header_sha256':sha(R/'src/llama-kv-cache-dsv4.h'),")
(P/'native.py').write_text(s);compile(s,str(P/'native.py'),'exec');shutil.copy2('/tmp/prepare-deferral-native.py',P/'prepare-native.py');print('Native OFF/ON/OFF controller prepared; it requires completed exact precision gates before execution.')
