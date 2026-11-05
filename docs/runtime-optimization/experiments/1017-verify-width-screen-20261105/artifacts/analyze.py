from pathlib import Path
import json,statistics,hashlib,subprocess,shutil,re,ast
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-verify-width-screen-20261105';P=E/'artifacts';run=json.loads((P/'results.json').read_text());assert run['complete']and run['all_outputs_state_splits_exact'];arms={k:json.loads((P/k/'results.json').read_text())for k in ['native','counts']};rows=[]
for nc,cc in zip(arms['native']['cases'],arms['counts']['cases']):
 assert nc['prefix']==cc['prefix']and nc['base_state_sha256']==cc['base_state_sha256']and nc['fixed_proposal_ids']==cc['fixed_proposal_ids'];prefix=nc['prefix']
 for n in range(1,7):
  a=[r for r in nc['rows']if r['queries']==n and r['kind']=='measured'];b=[r for r in cc['rows']if r['queries']==n and r['kind']=='measured'];assert len(a)==len(b)==10
  assert all(r['graphs_reused_delta']==1 for r in a+b);assert len({r['logits_sha256']for r in a+b})==1;assert len({r['splits']for r in a+b})==1
  med=statistics.median(r['us']for r in a);cm=statistics.median(r['us']for r in b);counts={key:statistics.median(r['counts'][key]for r in b)for key in b[0]['counts']};cold=[{'pass':r['pass'],'rep':r['rep'],'instantiate':r['counts']['cuda_graph_instantiate'],'us':r['us']}for r in b if r['counts']['cuda_graph_instantiate']]
  rows.append(dict(prefix=prefix,queries=n,native_median_us=med,native_mean_us=statistics.mean(r['us']for r in a),native_range_us=[min(r['us']for r in a),max(r['us']for r in a)],native_pass_medians_us=[statistics.median(r['us']for r in a if r['pass']==pas)for pas in [0,1]],counter_median_us=cm,counter_median_overhead_percent=100*(cm/med-1),counts_medians=counts,measured_cold_graph_rows=cold,all_reused=True,first_row_max_abs_vs_width1=max(r['first_row_max_abs_vs_width1']for r in a+b),logits_sha256=a[0]['logits_sha256'],target_state_sha256=next(r['target_state_sha256']for r in a if 'target_state_sha256'in r),splits=a[0]['splits']))
cliffs=[]
for prefix in [8192,90112]:
 a=next(r for r in rows if r['prefix']==prefix and r['queries']==5);b=next(r for r in rows if r['prefix']==prefix and r['queries']==6);cliffs.append(dict(prefix=prefix,width5_ms=a['native_median_us']/1000,width6_ms=b['native_median_us']/1000,extra_ms=(b['native_median_us']-a['native_median_us'])/1000,increase_percent=100*(b['native_median_us']/a['native_median_us']-1)))
mem={a['label']:{ph:{k:v['min_free']/2**20 for k,v in a['memory'][ph].items()if isinstance(v,dict)}for ph in ['load','inference']}for a in run['arms']};assert min(v for a in mem.values()for ph in a.values()for v in ph.values())>=512;assert all(a['exit_code']==0 and not a['warnings']and not a['floor_breach']for a in run['arms'])
sha=lambda f:hashlib.file_digest(Path(f).open('rb'),'sha256').hexdigest()
for f,h in run['source_hashes'].items():assert sha(R/f)==h
for f,h in run['config']['source_library_hashes'].items():assert sha(f)==h
assert sha(run['build']['private_library'])==run['build']['private_library_sha256'];assert not subprocess.check_output(['git','diff','--','src'],cwd=R,text=True)
# Preserve the exact reviewed source regions that explain the existing boundary.
regions={'src/llama-graph.cpp':[(1798,1879)],'src/models/deepseek41.cpp':[(1580,1620)],'ggml/src/ggml-cuda/mmvf.cu':[(620,656)],'ggml/src/ggml-cuda/mmvq.cu':[(290,329),(1208,1247)]};snapshots=[]
for i,(name,bounds)in enumerate(regions.items()):
 lines=(R/name).read_text().splitlines();text=''.join(f'\n## {name}:{start}-{end}\n'+''.join(f'{j+1}: {lines[j]}\n'for j in range(start-1,end))for start,end in bounds);f=P/f'guard-source-{i}.txt';f.write_text(text);snapshots.append({'source':name,'sha256':sha(R/name),'snapshot':f.name,'ranges':bounds})
summary={'complete':True,'head':run['head'],'prefixes':[8192,90112],'context':92160,'n_rs_seq':5,'measured_per_width_prefix_arm':10,'warmups_per_width_prefix_arm':3,'observations_per_arm':156,'observations_total':312,'sampled_target_state_checkpoints_per_arm':24,'native_counter_outputs_states_splits_exact':True,'same_width_repeats_exact':True,'base_restore_exact':True,'all_measured_llama_graphs_reused':True,'all_first_rows_exact_across_widths':all(r['first_row_max_abs_vs_width1']==0 for r in rows),'rows':rows,'cliffs':cliffs,'minimum_free_mib':mem,'source_regions':snapshots,'production_source_libraries_models_unchanged':True,'runtime_changes':False,'scope':'Fixed-prefix raw-logit target calls only. Restore, bridge, SHA and state checks excluded from decode+sync timing. Counter process distinct, no callback/CUPTI/backtrace/per-API clock or JSON. API graph/operation counts are not physical kernels, DMA or critical-path shares. Some width1/2/3/5 measured rows instantiate graphs after shape changes; retained and listed, medians used. Width6 measured rows have0 instantiations. Not a native server TG or acceptance/quality comparison.'};(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
table='\n'.join(f"| {n} | "+' | '.join(f"{next(r for r in rows if r['prefix']==pref and r['queries']==n)['native_median_us']/1000:.3f}"for pref in [8192,90112])+' |'for n in range(1,7))
ops=[]
for pref in [8192,90112]:
 for n in [5,6]:
  r=next(r for r in rows if r['prefix']==pref and r['queries']==n);c=r['counts_medians'];ops.append(f"| {pref} | {n} | {int(c['graphs'])} | {int(c['nodes'])} | {int(c['float_matmul'])} | {int(c['moe_matmul'])} | {int(c['cuda_graph_launch'])} | {int(c['event_wait'])} |")
extra='; '.join(f"{c['prefix']}: +{c['extra_ms']:.2f} ms / +{c['increase_percent']:.2f}%"for c in cliffs)
report=f'''# DeepSeek41 verification widths1..6, 2026-11-05

Completed read-only diagnostic after runtime commit3854c3785. The sixth query costs materially more than the fifth: {extra}. The observed branch expands MoE dispatch and strict floating per-query matmuls; extra graph construction/instantiation does not explain the width6 cliff. No runtime source/default/preset/deployment changed in this diagnostic.

## Conditions and timing definition

Same new-host Threadripper PRO3945WX/12threads/192GB/recorded six-GPU UUID order, Q3_K-AntirezLike target with MXFP8 Engram, Q8_0 KV, TopK2048, context92160, B8192/U512/P1, all15360experts on GPU with frozen overrides, expert cache0/staging0. Same BF16-named mixed-weight DSpark model loaded and feature/tail registration active; K5 and n_rs_seq5 retained for every width. New prefill deferral explicitly0; CPU lookup early unset/default ON. Existing private libllama from the preceding validated prototype is loaded in both arms, with unchanged production GGML modules. Full argv/env/source/model/binary identities: [run manifest](artifacts/results.json), [build](artifacts/build.json).

At prefix8192 and90112, use the same six fixed input token IDs truncated to each width1..6, without sampling/proposal selection. Before each target call restore identical sequence state and synchronize, outside the timer. Time llama_decode plus llama_synchronize; collect logits/hashes/state afterward. Prefill and feature processing are outside these intervals. Backend sampling is disabled so all requested logits are available. This includes raw-logit output handling and is not the native server GPU-sampling path, total TG or a DSpark acceptance benchmark.

Two fresh processes: clean target-call timing with no interposer, then a counter-only process. The latter increments counters while target decode+sync is armed; it adds no eval callback, CUPTI, backtrace, per-API timestamps, numeric reads or per-call JSON. A separate memory monitor enforces512MiB sampled free on every GPU. Two passes traverse widths in opposite order, each with five measured repetitions; first pass has two warmups, second one. All156observations per arm are retained,312total. Counters are diagnostic; timing below comes only from the uninstrumented process.

## Uninstrumented target call medians

| Queries (anchor included) | Prefix8K, ms | Prefix88K, ms |
| --- | ---: | ---: |
{table}

Each cell uses ten measured calls. Ranges, pass medians and counter overhead are in [summary](artifacts/summary.json). These medians are not an estimate of total server TG: no draft forward, proposal selection, feature bridge, acceptance/rejection decision or complete request is measured here. The six proposed IDs are fixed inputs, not fresh DSpark proposals.

## Operation and dispatch counts

| Prefix | Queries | Backend graph submissions | GGML nodes | Floating MUL_MAT | MUL_MAT_ID | CUDA Graph launch API | Event wait API |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(ops)}

MUL_MAT_ID expands120->720, exactly three expert projections x40layers x6separate query graphs instead of one grouped token batch. Floating matmuls expand275->950. Quantized ordinary MUL_MAT remains331 in this cohort. This identifies existing shape/dispatch fragmentation, not proof that any proposed implementation will achieve the full extra-cost reduction. API counts do not measure physical kernels, transfer bytes, wait durations or a unique additive share of runtime.

All measured llama graphs report reused_delta1 and allocation_graph0. Width6 has zero CUDA Graph instantiations in every measured counter row. After changing shapes in the reverse pass, one of ten width5calls instantiates39graphs on8K/38on88K; isolated width1/2/3cold rows also exist. They are preserved, not removed. Medians and both pass medians show the same large six-query cliff; cold rows cannot explain width6 being slower. An earlier progress message saying no measured instantiations existed was too broad; this paragraph and the raw counters give the actual scope.

## Existing source boundaries

- is_dsv41_verify_batch(kv_only=false) permits2..5queries; its KV-wide branch already permits6. Six queries with n_rs_seq5 are decode/verification, not prefill.
- build_dsv41_mm uses a4Dsample view for guarded floating batched GEMV through width5; width6 falls back to individual strict per-query floating matmuls and concatenation. Its quantized sample path already uses the six-query KV-wide eligibility.
- DeepSeek41 MoE grouping uses the generic five-query predicate; width6 calls the existing moe function six times. In the frozen tensor layout the first five layers have Q3_K/Q3_K/Q4_K expert weights, the other35layers Q3_K/Q3_K/Q3_K.
- GGML_CUDA_MMVF_SAMPLE_BATCH shares weight loads only through five samples. Existing generic six-sample kernels are present, but enabling their shared-load selection is a separate candidate from graph grouping.
- The current Ada MMVQ thresholds admit Q3_K through6columns and Q4_K through7; the current Ampere path admits this cohort. That source fact does not replace six-query numeric/rollback verification or establish Q2/Q4-model, other-GPU or CPU-cache support.

[Exact source regions](artifacts/summary.json) preserve source SHA256 and line-numbered excerpts. The next candidate should isolate grouped MoE6 from floating sample-grid6, without globally changing every caller of the generic max_tokens predicate or enabling F16 compact attention unintentionally.

## Integrity, memory and limits

All same-width repeated full logit vectors match. Target state hashes captured at the last repetition of each pass also match (24 checkpoints per arm), and base-state restore is exact for both prefixes. Native/counter logits and split counts match for all 156 observations; sampled per-width states, base snapshots and fixed IDs match separately. First-position logits are also exact across widths1..6. This is benchmark integrity of unchanged code, not new-model quality proof or an all-position serial-vs-batch gate.

Both processes exit0 without forced kill, allocation-size mismatch, CUDA/OOM errors or floor breach. Sampled minimum free: native1397.125MiB, counters1391.125MiB. Source/modules and target model stat identities are unchanged; no processes remain on GPU after completion. Original code commit remains3854c3785. [Harness](artifacts/width.cpp), [counter source](artifacts/counts.cpp), [controller](artifacts/run.py), [analysis](artifacts/analyze.py), [logs](artifacts/run.log). Initial unexecuted harness/build snapshots are retained; before launch only incremental observation persistence was added to preserve partial evidence on failure.

## Next bounded work

Prepare independent opt-in/defaultOFF graph extensions: first grouped MoE for the sixth query using existing expert kernels, then floating sample-grid6. Retain old5query/other-mode fallbacks, Q8/text/P1/full-GPU scope, expert IDs, weights, precision/reduction order, causal masks, taps, recurrent state and rollback. Any combined candidate needs full logits/taps/KV/state/NLL/rejected-window replay gates and fresh native same-K OFF/ON/OFF before adoption. Shared-weight MMVF6 selection is a later, separately measured kernel step. No default promotion or promised TG gain from this diagnostic.
''';(E/'REPORT.md').write_text(report);shutil.copy2('/tmp/analyze-verify-width-screen.py',P/'analyze.py');print('SCREEN_COMPLETE',extra);print('cold rows',sum(len(r['measured_cold_graph_rows'])for r in rows));print('native counter median overhead range',min(r['counter_median_overhead_percent']for r in rows),max(r['counter_median_overhead_percent']for r in rows))
