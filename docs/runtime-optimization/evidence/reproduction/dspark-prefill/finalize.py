from pathlib import Path
import json,hashlib,re,statistics,shutil,subprocess
P=Path(__file__).resolve().parent;R=Path('/home/sergei/Github/llama.cpp');D=Path('/home/sergei/_my_old/_my_deepseek41');F=D/'validation/moe-cache/1017-dspark-prefill-skip-20261103';A=F/'artifacts'
b=json.loads((P/'bench-results.json').read_text());assert b['complete'] and len(b['summary'])==2
assert b['patch']==subprocess.check_output(['git','diff'],cwd=R,text=True)
assert json.loads((P/'long-summary.json').read_text())['complete']and json.loads((P/'transport-summary.json').read_text())['complete']
mods=[];comps=[];memory={};ranges={};acceptance={}
for arm,a in b['arms'].items():
 assert a['complete']and a['exit_code']==0 and not a['warnings']and not a.get('forced_kill')and len(a['requests'])==8
 env=a['environment'];assert 'LD_PRELOAD'not in env and not any(k.startswith('DSV41_')for k in env)and env['LLAMA_DSPARK_PROFILE']=='0'
 log=(P/f'bench-{arm}/server.log').read_text(errors='replace');assert not re.search(r'\bE ',log)
 assert 'GPU feature bridge = enabled'in log and 'producer-local features enabled'in log
 assert 'GPU feature packing fallback to host'not in log and 'using host features'not in log and 'DSPARK_PROFILE stage='not in log
 mods.append([l.split('load_tensors:',1)[1]for l in log.splitlines()if'load_tensors:'in l and('model buffer size'in l or'offload'in l)])
 comps.append(re.findall(r'((?:CUDA\d+|CUDA_Host|CPU)\s+compute buffer size\s*=\s*[^\r\n]+)',log))
 memory[arm]=min(v['min_free']for v in a['memory']['inference'].values()if isinstance(v,dict)and'min_free'in v)/1048576;assert memory[arm]>=512 and not a['memory'].get('error')
 ranges[arm]={};acceptance[arm]={}
 for n in [8192,98304]:
  qs=[q['response']['timings']for q in a['requests']if q['n']==n and not q['warmup']];assert len(qs)==3
  ranges[arm][n]={k:[min(q[k]for q in qs),max(q[k]for q in qs)]for k in ['prompt_per_second','predicted_per_second']}
  acceptance[arm][n]={'draft_n':qs[0]['draft_n'],'draft_n_accepted':qs[0]['draft_n_accepted'],'rate':qs[0]['draft_n_accepted']/qs[0]['draft_n']}
assert mods[0]and mods[0]==mods[1]and comps[0]and comps[0]==comps[1]
e0=b['arms']['off']['environment'];e1=dict(b['arms']['on']['environment']);assert e1.pop('LLAMA_DSV41_PREFILL_SKIP')=='1'and e0==e1
assert b['arms']['off']['command']==b['arms']['on']['command']
for n in [8192,98304]:
 qs=[q for a in b['arms'].values()for q in a['requests']if q['n']==n];assert len(qs)==8 and len({q['tokens_sha256']for q in qs})==1
 assert len({tuple(q['response']['timings'][k]for k in ['draft_n','draft_n_accepted'])for q in qs})==1
assert not b['arms']['off']['skip_lines']and not b['arms']['off']['seed_lines']and len(b['arms']['on']['skip_lines'])==len(b['arms']['on']['seed_lines'])==52
s={'complete':True,'head':b['head'],'speed':b['summary'],'ranges':ranges,'acceptance':acceptance['off'],'server_min_free_mib':memory,'tokens_and_counters_exact':True,'model_and_compute_buffers_exact':True,'request_count':16,'native_long':json.loads((P/'long-summary.json').read_text()),'transport':json.loads((P/'transport-summary.json').read_text())}
(P/'summary.json').write_text(json.dumps(s,indent=2)+'\n');(P/'final.patch').write_text(b['patch'])
for name in ['bench.py','bench-results.json','bench-driver.log','bench-off/server.log','bench-on/server.log','summary.json','final.patch','finalize.py','build-server.log']:
 q=A/name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/name,q)
rows='\n'.join(f"| {r['n']} | {r['off']['prompt_per_second']:.2f} | {r['on']['prompt_per_second']:.2f} | {r['pp_change_pct']:+.2f}% | {r['off']['predicted_per_second']:.2f} | {r['on']['predicted_per_second']:.2f} | {r['tg_change_pct']:+.2f}% |"for r in b['summary'])
table='''Медианы трёх измеренных повторов:

| Prefix | OFF PP,t/s | ON PP,t/s | ΔPP | OFF TG,t/s | ON TG,t/s | ΔTG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+rows
text=(F/'REPORT.md').read_text().replace('production-policy server A/B выполняется.','production-policy server A/B завершён.').replace('-36 rollback cases:','- 36 rollback cases:')
text=text.replace('Результат пока ожидается.',table+f'''

Все16 response token arrays совпали внутри/между arms для своего prefix; draft_n/draft_n_accepted также exact. На8K155/215 (72.09%), на98K154/188 (81.91%) в каждом запросе. НаON52 active skip+seed calls:44 logical8192/cut4608/first_valid7168 и8 logical7676/cut4096/first_valid6656. OFF0. Startup model/offload и compute buffer lines exact; environment отличается только M2 флагом, все library hashes одинаковы. LOCAL bridge включён в обоих arms, CPU packing fallback отсутствует. Оба exit0 без CUDA/OOM/allocation mismatch/forced kill. Actual minimum free VRAM OFF{memory['off']:.3f}MiB,ON{memory['on']:.3f}MiB>=512. GPU освобождены.

Один порядок OFF→ON и один frozen text workload ограничивают перенос результата; это не сравнение с Maya. Малые изменения TG не считать установленным ускорением/регрессией. Скорость8K/98K не прогнозируется из target-only +53.25%: здесь другой retained tail и настоящий BF16 drafter. Полные диапазоны повторов сохранены в summary.json.
''')
text+='\n[Итоговые результаты](artifacts/summary.json), [server raw results](artifacts/bench-results.json), [финальный patch](artifacts/final.patch), [контрольные суммы](artifacts/SHA256.json).\n';(F/'REPORT.md').write_text(text)
p=A/'PROTOCOL.md';p.write_text(p.read_text().replace('Final benchmark results pending.','Final results are in summary.json: all16outputs/counters and model/compute buffer lines match,52active skip/seed calls onlyON,clean exits and actual VRAM floor512. final.patch applies to309d6ca9b and reproduces the measured server build.'))
(A/'SHA256.json').write_text(json.dumps({str(p.relative_to(A)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sorted(A.rglob('*'))if p.is_file()and p.name!='SHA256.json'},indent=2)+'\n')
print(json.dumps({'speed':b['summary'],'server_min_free_mib':memory,'acceptance':acceptance['off']},indent=2))
