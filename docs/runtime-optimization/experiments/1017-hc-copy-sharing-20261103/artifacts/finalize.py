from pathlib import Path
import json,re,hashlib,subprocess,shutil,datetime
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp')
sha=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
cfg=json.load(open(P/'config.json'));b=json.load(open(P/'bench-results.json'));n=json.load(open(P/'long-summary.json'));s=json.load(open(P/'scheduler-counted-results.json'));m=json.load(open(P/'metadata-results.json'))
assert all(x['complete']for x in [b,n,s,m]);assert len(b['arms'])==3 and sum(len(a['requests'])for a in b['arms'])==27
assert s['capture_outputs_per_mode']==32 and len(s['cases'])==51
assert subprocess.check_output(['git','diff','--','ggml/src/ggml-backend.cpp'],cwd=R,text=True)==cfg['runtime_patch']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
for name,h in cfg['source_library_hashes'].items():assert sha(R/'build-aliceai/bin'/name)==h
out={'complete':True,'head':cfg['source_head'],'runtime_source_sha256':sha(R/'ggml/src/ggml-backend.cpp'),'default':'OFF','numerical':n,'micro':{'cases':51,'capture_outputs_per_mode':32,'eligible':[{k:x[k]for k in ['mode','steps','aliases','full_copy_requests']}for x in s['cases']if x['variant']=='eligible']},'speed':b['summary'],'arms':[]}
models=[]
for a in b['arms']:
 assert a['complete']and a['exit_code']==0 and not a.get('error')and not a.get('forced_kill')and not a['warnings']and not a['floor_hit']
 assert a['m2_active']and a['local_active']and not a['memory'].get('error')
 log=(P/('bench-'+a['label'])/'server.log').read_text(errors='replace')
 assert re.search(r'n_ctx_slot = 100352',log),a['label']
 placed=re.findall(r'load_tensors:\s+(CUDA\d+) model buffer size =\s+([0-9.]+) MiB',log);models.append(placed);assert len(placed)==8
 minimum=min(v['min_free']for phase in ['load','inference']for v in a['memory'][phase].values()if isinstance(v,dict)and'min_free'in v)/1048576
 assert minimum>=512
 out['arms'].append({'label':a['label'],'requests':len(a['requests']),'sampled_min_free_mib':minimum,'model_buffers':placed,'inference_free_by_uuid':{u:{'nvml_index':v['index'],'min_free_mib':v['min_free']/1048576}for u,v in a['memory']['inference'].items()if isinstance(v,dict)}})
assert models[0]==models[1]==models[2]
for width in [512,8192,98304]:
 rows=[q for a in b['arms']for q in a['requests']if q['n']==width]
 assert len({q['tokens_sha256']for q in rows})==1
 assert len({tuple(q['response']['timings'][k]for k in ['draft_n','draft_n_accepted'])for q in rows})==1
 assert all(q['response']['timings']['cache_n']==0 and len(q['response']['tokens'])==q['response']['timings']['predicted_n']==256 for q in rows)
meta_off=m['cases'][1]['result'];meta_on=m['cases'][2]['result'];deltas=[]
for x,y in zip(meta_off['memory'],meta_on['memory']):
 keys=['main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']
 saved=sum(x[k]-y[k]for k in keys)/1048576;deltas.append({'device':x['device'],'saved_mib':saved})
assert [x['saved_mib']for x in deltas]==[0,160,160,160,160,0]
out['combined_reserve_saved_mib']=640;out['reserve_by_device']=deltas
out['memory_scope']='Actual K3freshON process saves160MiB onfourcards.NumericalK5processstartsOFF,retainscapacitywhiletogglingON.Metadata176128ONonly;noactualON176128runhere.'
out['post_run']={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'compute_apps':'','gpu_memory_csv':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,memory.used,memory.free','--format=csv,noheader,nounits'],text=True)}
(P/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
(P/'MEASURED.patch').write_text(cfg['runtime_patch']);shutil.copyfile(R/'ggml/src/ggml-backend.cpp',P/'ggml-backend.after.cpp')
shutil.copyfile('/tmp/dsv41-m2-residual-profile-20261103/inputs.json',P/'bench-inputs.json')
shutil.copyfile('/tmp/dsv41-m2-large-context-20261103/gpu-map.json',P/'gpu-map-reference.json')
prov={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),'head':cfg['source_head'],'baseline_runtime':'6d038260e','measured_patch_sha256':sha(P/'MEASURED.patch'),'runtime_source_before_sha256':sha(P/'ggml-backend.before.cpp'),'runtime_source_after_sha256':sha(P/'ggml-backend.after.cpp'),'library_hashes':{f.name:sha(f)for f in (R/'build-aliceai/bin').glob('*.so')},'executable_sha256':sha(R/'build-aliceai/bin/llama-server'),'helper_locators':{name:{'path':str(P/name),'sha256':sha(P/name)}for name in ['metadata','long','trace.so','scheduler','scheduler-counted']},'model_payloads':'Recorded paths/metadata/signatures;weights not modified or fully rehashed in this stage.','scheduler_source':'scheduler.pre-counter.cpp is the final uncounted/capture harness source;scheduler.cpp adds owned-backend copy counters used by scheduler-counted. scheduler.log is earlier smoke before fixed-graph capture and is not final capture evidence.','mapping':'CUDA ordinal/PCI/UUID mapping from fresh same-env prior M2gate;NVML UUID/floor samples fresh for every currentprocess. No auxiliary CUDA map process during timings.','cleanup':'All own tests exit0/no forcedkill;post-runGPUlistempty.','no_promotion':'DefaultOFF.No new implementation commit or deploy.'}
(P/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
E=R/'docs/runtime-optimization/experiments/1017-hc-copy-sharing-20261103';A=E/'artifacts';A.mkdir(parents=True,exist_ok=True)
excluded={'long','trace.so','scheduler','scheduler-counted','metadata','patch.py','STATUS.md'}
for f in P.rglob('*'):
 if not f.is_file()or'__pycache__'in f.parts or f.name in excluded or f.suffix=='.pyc':continue
 dest=A/f.relative_to(P);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
lines=[]
for row in b['summary']:
 x=row['arms'];refpp=(x['off-pre']['PP']+x['off-post']['PP'])/2;reftg=(x['off-pre']['TG']+x['off-post']['TG'])/2
 lines.append(f"| {row['n']} | {refpp:.2f} | {x['on']['PP']:.2f} | {row['PP_delta_pct']:+.2f}% | {reftg:.2f} | {x['on']['TG']:.2f} | {row['TG_delta_pct']:+.2f}% |")
report='''# HC root/reshape sharing: prototype and validation, 2026-11-03

**Прототип реализован и проверен: PP +3.8..4.3%, combined GPU reserve -640MiB.** Default `GGML_SCHED_COPY_RESHAPE` остаётся OFF. Новый код и результаты не закоммичены; base my_maya9b74cc6e6/runtime6d038260e плюс [measured patch](artifacts/MEASURED.patch). Это продолжение согласованного [дизайна](../../designs/1017-hc-copy-sharing.md), не полный перенос Unsloth/Maya и не новое сравнение с ними.

## Реализация

Меняется только ggml/src/ggml-backend.cpp: protected pipeline slots/n_copies>1, CUDA->CUDA, direct full zero-offset contiguous F32 reshape от HC_POST. Canonical destination root уже должен быть input текущего split и иметь INPUT/OUTPUT protection. Его transfer/events/overwrite guards остаются. Только не-inplace RMS_NORM consumer получает destination view; allocation-only view вставляется перед первым consumer в graph_copy для корректного n_views. Нет глобального alias entry между split, ручной подмены data pointer, нового GPU buffer/kernel/stream, изменений allocator reuse policy или compute_splits.

Read-only prepass отклоняет roots с неизвестными/пишущими consumers, неподходящими views и input/export flags. Callback, CPU/другой backend, n_copies1, nested/offset/subrange/noncontiguous views и поздний/другой root используют старый путь. Dtype F32 задан валидным API HC_POST/HC_PRE; искусственный F16 HC_POST numerical test не создавался. [Протокол](artifacts/PROTOCOL.md), [source до/после](artifacts/ggml-backend.after.cpp).

## Проверки точности и lifetime

| Gate | Результат |
| --- | --- |
| Малый CUDA0/CUDA1/CPU scheduler | 17 variants xOFF/ON/default =51 сочетание; exact outputs и правильный alias/fallback |
| Copy-slot rotation | 160 eligible chunks/mode, widths1/4/6/7/508/511/512/513; delayed host outputs exact |
| Число transfer requests | На160 chunks OFF/default320, ON160; actual destination storage shared только ON |
| Capture/replay | По32 меняющихся input/mode; на каждой GPU ON capture1/launches31, update failures0; outputs exact к OFF |
| Compute failure/retry micro | По одной ошибке в каждом OFF/ON/default режиме, последующие outputs exact |
| Full target+draft | Два полных98K corpora,36 result groups OFF/ON/OFF; target/draft logits, seq/global/partial states и taps exact |
| Restore/append | Full/seq/partial checkpoint,90K resume,7676+512+4 tail; append1/7/511/513 exact |
| Speculation/rollback | Raw K3/K5 IDs/confidence/logits;36 zero/partial/full acceptance rollback workflows и следующий draft exact |
| Seed compute failure | Две контролируемые ошибки, очистка обеих state/range и full reload/retry exact |
| NLL text/code | По128 labels/corpus/mode: обе deltaNLL=0, PPL ratio1; raw logits/top1 exact |

Full helper: Q3_K-AntirezLike/MXFP8 Engram/Q8KV/BF16 DSpark,ctx100352/B8192/U512/P1/TopK2048/frozenallGPU/fit-cache-staging-mmprojOFF. K5/RS5/output6,p_min0/backend samplingOFF нужны для raw logits/confidence. M2/default LOCAL ON, ASYNC OFF. Observer подтверждает1836 shared RMS inputs только ON,0 OFF; M2 активен во всех tested paths. Native minimum free1325.25MiB>=512,exit0. Expected E-логи находятся только внутри двух FAULT_BEGIN/END; неожиданных ошибок нет.

Это transport-only gate относительно текущего runtime.128 labels на корпус не являются новой общей оценкой качества модели; согласованный предел +0.01nat/token выполнен точно. Source всех probes, raw states/logits hashes, fault results и library identity сохранены в [long summary](artifacts/long-summary.json), [raw result](artifacts/long-native/results.json), [micro result](artifacts/scheduler-counted-results.json).

## Clean native speed

Три отдельных server processes OFF-pre/ON/OFF-post, тот же ctx100352/layout/Q8KV, BF16K3/full/p_min0.5/backend samplingON, M2/default LOCAL, ASYNC OFF. Никаких preload/profiler/callbacks/calibration/builds в timed run. Full uncached raw input IDs,256greedy/seed1234/ignore_eos.512warm+2 measured,8Kwarm+3,98Kwarm+1 на каждом процессе. Всего27 запросов; полные output arrays и draft generated/accepted counters exact для каждого prefix во всех arms, включая warmup.

OFF ниже - среднее двух OFF-медиан, ON - медиана своего arm:

| Prefix | OFF PP,t/s | ON PP,t/s | DeltaPP | OFF TG,t/s | ON TG,t/s | DeltaTG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
'''+ '\n'.join(lines)+'''

PP OFF drift между первым и последним control: -0.11% на512,-0.44% на8K,+0.19% на98K. Диапазоны PP8K OFF-pre1043.20..1046.61, ON1082.76..1088.52, OFF-post1040.11..1042.62. Это поддерживает полезный PP gain в данном fixture. Один порядок процессов, один speed corpus и singleton measured98K ограничивают перенос вывода. 98K TG delta-0.50% - одиночная точка; для вывода о небольшой TG регрессии нужна отдельная серия. Заметного TG выигрыша этот опыт не установил.

Это полный engine PP/TG, не оценка из суммы memcpy durations. Новый runtime не профилировался Nsight в timed runs. [Historical staged-copy screen](../1017-hc-copy-design-20261103/REPORT.md) остаётся отдельным свидетельством пути, не новым attribution всего4% только PCIe. [Все request/arm data и ranges](artifacts/bench-results.json).

## Память, размещение и завершение

Шесть свежих no_alloc cases:100352/default/OFF/ON,176128/ON,100352/K5OFF/ON. Default/OFF budgets exact; model/KV и metadata per-tensor routes прежние,15360 routed GPU bundles, host routed bytes0. Это fixed-layout measure, не свободный auto-fit solver. Combined bound target+draft/taps reserve уменьшается по160MiB на CUDA1/2/3/4, суммарно640MiB. Standalone target component также уменьшается на CUDA5, но после binding taps37/38 этот выигрыш там исчезает;800MiB combined не заявляется.

В fresh ON production process NVML подтвердил +160MiB free на четырёх картах относительно OFF. Global sampled minimum почти прежний из-за другой bottleneck GPU: OFF-pre1435.25MiB, ON1439.25MiB; все arms выше512. В K5 numerical process начальные OFF capacities удерживаются при переключении ON, поэтому он не является тестом уменьшения fresh ON capacity. Actual model buffers target и sidecar совпали во всех трёх arms; per-tensor parity проверена в metadata. Отдельного loader observer в clean timed processes не было.

NVML sampling100ms, stop guard200ms только для owned process; это sampled floor, не гарантия отсутствия пика между samples. Все процессы exit0, без OOM/CUDA error/compute-size mismatch, forced kill или floor breach. После опыта GPU compute apps отсутствуют. Actual176128 с новым flagON не запускался: там выполнен только no_alloc; прежний M2 actual176128 относится к copy-sharing OFF. [Подробности памяти](artifacts/summary.json), [provenance](artifacts/provenance.json).

## Решение и использование

Оставлен bounded opt-in кандидат: prefill gain полезен, точность и lifetime gates пройдены. Default OFF сохраняется по согласованному дизайну. Включение в тестовой сборке:

```bash
GGML_SCHED_COPY_RESHAPE=1 ./llama-server ...
```

Unset или0 возвращает исходные копии. Это runtime environment variable, не INI/CMake option. Нет автоматического изменения пользовательских presets/my_build, model conversion, commit, default promotion или merge.

Другие кванты/LLM, F16KV, parallel>1/media/LoRA, cache/spill и произвольные view-sharing режимы этим набором не закрыты. Следующие отдельные пункты: dense idx_score/candidate_mask traffic, matched Maya и Q2/Q4 MMQ. Для нового расширения alias scope сначала нужен новый lifetime/graph design. [Артефакты и checksums](artifacts/SHA256.json).
'''
report=report.replace('OFF-pre1435.25MiB, ON1439.25MiB;', 'OFF-pre1435.25MiB, ON1439.25MiB, OFF-post'+str(out['arms'][2]['sampled_min_free_mib'])+'MiB;')
(E/'REPORT.md').write_text(report)
manifest={str(f.relative_to(A)):sha(f)for f in sorted(A.rglob('*'))if f.is_file()and f.name!='SHA256.json'}
(A/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('SAVED',E,len(manifest),'artifacts')
print('SUMMARY',[(x['n'],x['PP_delta_pct'],x['TG_delta_pct'])for x in b['summary']])
print('MEMORY',[(a['label'],a['sampled_min_free_mib'])for a in out['arms']])
