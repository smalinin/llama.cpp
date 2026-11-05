from pathlib import Path
import json,hashlib,subprocess,statistics,shutil,re
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-maya-newhost-20261105';P=E/'artifacts'
def sha(f):
 with f.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
def main():
 subprocess.run(['python3',str(P/'summarize.py')],stdout=(P/'summarize.log').open('w'),check=True)
 j=json.loads((P/'results.json').read_text());s=json.loads((P/'summary.json').read_text());c=j['config'];assert s['complete']
 for path,h in c['source_library_hashes'].items():assert sha(Path(path))==h,path
 for path,h in c['source_files'].items():assert sha(R/path)==h,path
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=20).strip()
 old=json.loads((P/'pre-build-hashes.json').read_text());original=Path('/media/sergei/WW2T/_maya/project-maya-1.1.26/build/maya-ds4');assert sha(original)==old[str(original)]
 table=[];cpu=[];limits=[]
 for mode in ['off','on']:
  for n in [512,8192,98304]:
   a=next(x for x in s['rows'] if x['label']=='ours-'+mode and x['prefix']==n);b=next(x for x in s['rows'] if x['label']=='maya-'+mode and x['prefix']==n);delta=next(x for x in s['comparisons'] if x['sidecar']==mode and x['prefix']==n)
   fmt=lambda v:'-' if v is None else f'{100*v:.2f}%'
   table.append(f"| {mode} | {n} | {a['pp']:.2f} | {b['pp']:.2f} | {delta['pp_maya_delta_pct']:+.2f}% | {a['tg']:.2f} | {b['tg']:.2f} | {delta['tg_maya_delta_pct']:+.2f}% |")
   cpu.append(f"| {mode} | {n} | {a['cpu_s']:.2f} | {b['cpu_s']:.2f} | {a['mean_cpu_cores']:.2f} | {b['mean_cpu_cores']:.2f} | {fmt(a['acceptance'])} | {fmt(b['acceptance'])} |")
   limits.append(f"| {mode} | {n} | {a['unique_output_sequences']}/{a['repeats']} | {b['unique_output_sequences']}/{b['repeats']} | {a['pp_range'][0]:.2f}..{a['pp_range'][1]:.2f} | {b['pp_range'][0]:.2f}..{b['pp_range'][1]:.2f} |")
 memory='\n'.join(f'| {label} | {v:.2f} |' for label,v in s['minimum_free_mib'].items())
 report='''# Наш runtime и Maya: новое оборудование, 2026-11-05

**Серия завершена: четыре свежих процесса, 52 полных uncached запроса, exit 0.** Материнская плата, CPU, RAM и PCIe-порядок GPU изменились. Все таблицы ниже относятся только к новой машине. Скорости, калибровки и допуски памяти прежних серий не использовались как baseline. Runtime source и defaults не менялись; commit/push не выполнялись.

## Новая платформа и binaries

Threadripper PRO 3945WX, 12 ядер/24 потока, 192 GB RAM, одна NUMA node. Оба движка получили 12 CPU threads и одинаковый PCI-порядок GPU через явные UUID: пять карт 48 GB, последняя RTX 3090 24 GB. [Hardware snapshot](artifacts/lscpu.txt), [RAM](artifacts/meminfo.txt), [GPU UUID/PCI](artifacts/gpu.txt), [topology](artifacts/topology.txt), board/BIOS files в artifacts. GPU labels прежней платы не переиспользуются как адреса устройств.

Исходники нашего runtime: `my_build 91c88beb2`. Сервер/CUDA унаследованы из проверенной сборки; embedded build-info может показывать предыдущий parent `b3377774b`. Точные source и library hashes сохранены в [config](artifacts/config.json). Наш CPU backend пересобран с AVX2/FMA/F16C, GGML_NATIVE=OFF. CPU-объекты Maya ggml пересобраны с `-march=native` на новом CPU; ISA AVX2 подтверждена, AVX512 отсутствует. Использован отдельный Maya executable с прежними CUDA objects и новой CPU library. Пользовательский `build/maya-ds4` не изменился. CMake logs/caches, CPU flags, native macros, compiler version и private link argv сохранены. Это замороженные binaries данной серии, не заявление об одинаковых compiler flags движков.

## Протокол

Оба движка читают одни десять SSD GGUF Q3_K-AntirezLike, включая MXFP8 Engram. Maya pack заново построен прямо из этих GGUF; weights не копировались и не менялись. Одинаковый sidecar по пути `dspark-DeepSeek-V4.1-Flash-1017-BF16.gguf`: фактически 7.219 GB payload MXFP4, 1.262 GB BF16 и 0.020 GB F32, а не однородный BF16. [Типы](artifacts/tensor-types.json), SHA256 sidecar и stats каждого shard в config.

Одинаковые входные token IDs technical-prose, длины 512/8192/98304, context=100352, TopK=2048, chunk/ubatch=512, 256 greedy output tokens, temperature=0/top_k=1/top_p=1/min_p=0/repeat_penalty=1/seed=1234, EOS игнорируется. Между запросами sentinel token 97. У всех запросов server cache_n=0 / Maya reused=0, prefix прочитан полностью. TG вычислен по **255 decode intervals** у обоих движков: первый токен относится к prefill.

Порядок: наш OFF, Maya OFF, Maya ON, наш ON. На 512/8K по два warmup и три measured; на 98K один warmup и два measured. В таблицах медианы measured, все warmups и ranges сохранены. В native timing нет profiler/interposer/eval callback, сборки или параллельной калибровки.

Sidecar ON: max=5, individual confidence threshold=0.5, cumulative admission=0, empty cooldown=1; у нас verify policy full, backend sampling согласно frozen args. Наш runtime: B8192/U512, Q8_0 K/V, фиксированные GPU split/overrides, expert cache=0 и prefill staging=0. Это прежняя геометрия размещения, заново проверенная actual загрузкой и памятью; старые measured memory estimates не использованы. Maya: собственные KV и layer/expert placement, TP1/threads ON/reserve1024/mmap-experts auto/vision OFF. Старые lane-times не копировались; CPU/PCIe-калибровка снята заново в каждом private pack. Скопирован только профиль модельных частот экспертов для hot-start, без hardware timings. Исходный Maya pack/profile/config не изменён, hashes проверены.

## Скорость новой серии

Delta означает Maya / наш runtime - 1. Единицы PP/TG: tokens/s.

| Sidecar | Prefix | Наш PP | Maya PP | Delta PP | Наш TG | Maya TG | Delta TG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
TABLE

Это end-to-end сравнение движков с разными KV, arithmetic и placement. Оно не изолирует эффект одного kernel или expert cache. На разных autoregressive продолжениях TG также зависит от маршрутов экспертов и поведения drafter; одинаковые input IDs не делают output workload идентичным.

## CPU и acceptance

CPU seconds - Linux process user+system time всех threads за полный запрос, включая PP/TG и обработку protocol. Mean CPU cores = CPU seconds / wall seconds; это не доля CPU expert math и не decode-only профиль. Acceptance - сумма accepted / offered по measured запросам, не среднее процентов.

| Sidecar | Prefix | Наш CPU,s | Maya CPU,s | Наш cores | Maya cores | Наш acceptance | Maya acceptance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
CPU

Больший расход CPU не доказывает лучшее исполнение CPU FFN. В него входят workers, dispatch и ожидания; их вклад эта серия не разделяет. Per-position kept/reached Maya сохранены в raw log и results. Acceptance на разных generated prefixes не является прямой приёмкой качества drafter bridge; для неё нужны paired logits/drafts на одинаковом prefix.

## Память и размещение

У нашего target GPU-only expert overrides и offload 41/41; routed experts размещены на GPU. CPU_Mapped buffers также имеются, включая Engram и mapped views плотного shard; размер mapped buffer не равен объёму CPU expert compute. Полные model buffer строки сохранены. У Maya без sidecar 92 target experts на CPU lane (15268 target experts в GPU pool), с sidecar 153 (15207 target experts в GPU pool). Drafters добавляют свои experts; нельзя считать общие group counts вместе с MTP как число target experts. Финальные per-GPU layout и STAT tiers сохранены в results/logs. Общий resident payload и размещение между движками не идентичны.

| Arm | Sampled minimum free VRAM, MiB |
| --- | ---: |
MEMORY

Все четыре процесса завершились с exit 0, без recorded CUDA errors, allocation mismatch/OOM, forced kill или нарушения 512 MiB floor. NVML sample 100 ms / watcher 200 ms не исключают пик между samples. Контекст 100352 не является тестом 262K и не задаёт универсальный memory limit. Старые no_alloc/actual-context результаты с другой машины сюда не перенесены.

## Повторы и границы

Число уникальных output sequences / число measured повторов и PP ranges:

| Sidecar | Prefix | Наш unique/runs | Maya unique/runs | Наш PP range | Maya PP range |
| --- | ---: | ---: | ---: | --- | --- |
LIMITS

Серия измеряет скорость, не NLL/PPL, logits/state/rollback correctness. Отличие повторных greedy output требует отдельной диагностики и не даёт объявить движок принятым по качеству. Причины различий, включая CPU/GPU rounding и placement/cache, по одним timings не определяются. Фиксированный K5/full/cumulative=0 не является подбором лучшего n_max или adaptive policy для каждого движка. Один arm order, малое число повторов, особенно два на 98K, и разные generated continuations ограничивают выводы о малых deltas. Исторические проценты не складываются и не сравниваются с этой серией.

## Дальнейшие шаги

Следующий этап - свежий профиль prefill 8K/98K и target verification на этой платформе: GPU overlap между chunk512, host waits/dispatch, routed MMQ/attention/Engram и упаковка KV. Любой новый кандидат должен пройти свежие native speed/memory и согласованные logits/state/rollback/NLL gates на новой машине. Этот отчёт не объявляет перенос Maya целиком или изменение defaults.

[Полные результаты](artifacts/results.json), [сводка](artifacts/summary.json), [очередь](artifacts/run.log), [runner](artifacts/run-comparison.py), [подготовка](artifacts/prepare.py), [SHA256](artifacts/SHA256.json). Compiled private Maya binary локально в artifacts/bin и исключён из Git; его hash и link argv сохранены. Все собственные GPU-процессы завершены.
'''
 report=report.replace('TABLE','\n'.join(table)).replace('\nCPU\n','\n'+'\n'.join(cpu)+'\n').replace('\nMEMORY\n','\n'+memory+'\n').replace('\nLIMITS\n','\n'+'\n'.join(limits)+'\n');(E/'REPORT.md').write_text(report)
 (P/'STATE.md').write_text('# Completed new-host comparison\n\n52 full requests; all four exits 0, fresh new-host calibration and hardware mapping. No historical speed baseline. Source/defaults unchanged. See ../REPORT.md and summary.json for bounds. Next: fresh prefill/verification profile. No commit/push performed.\n')
 for doc,rel in [(R/'docs/runtime-optimization/CATALOG.md','experiments/1017-maya-newhost-20261105/REPORT.md'),(R/'docs/runtime-optimization/models/deepseek41-1017.md','../experiments/1017-maya-newhost-20261105/REPORT.md')]:
  doc.write_text(doc.read_text()+'\n\n## Новая платформа, 2026-11-05\n\n[Новая независимая серия]('+rel+'): Threadripper PRO 3945WX, 192 GB RAM, новый PCI/UUID-порядок GPU, CPU threads12. Все прежние timings и memory/cost estimates относятся к другой машине. Выполнено сравнение PP/TG без sidecar и с тем же sidecar на 512/8K/98K, с actual memory floor и fresh calibration. Результаты скорости, CPU, acceptance и repeat limitations приведены в отчёте. Далее свежий профиль prefill/verification; runtime/defaults не менялись.\n')
 n=Path('/home/sergei/_my_old/_my_deepseek41')
 for name in ['DEEPSEEK_V41_1017_MAYA_CACHE_STAGE5_DESIGN.md','DEEPSEEK_V41_1017_MAYA_EXPERT_CACHE_PLAN.md']:
  f=n/name;f.write_text(f.read_text()+'\n\n## Новая машина: 2026-11-05\n\nCPU/материнская плата/RAM/PCIe-порядок GPU изменены. Старые результаты не являются baseline текущей платформы. [Независимое повторное сравнение]('+str(E/'REPORT.md')+') завершено: 52 full uncached запроса, четыре fresh процесса OFF/ON sidecar, 512/8K/98K; новые CPU builds, GPU mapping и lane calibration, source/defaults прежние. Далее fresh prefill/target verification profile на этой машине. Quality/NLL/state gates этим speed-опытом не заменены.\n')
 (P/'SHA256.json').write_text(json.dumps({str(f.relative_to(P)):sha(f) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='SHA256.json' and 'bin' not in f.relative_to(P).parts and '__pycache__' not in f.relative_to(P).parts},indent=2)+'\n');print('REPORT_WRITTEN',E/'REPORT.md')
if __name__=='__main__':main()
