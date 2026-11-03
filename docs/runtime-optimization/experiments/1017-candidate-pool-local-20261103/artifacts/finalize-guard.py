from pathlib import Path
import json,subprocess,hashlib,os
P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');assert json.load(open(P/'fresh-on-summary.json'))['complete'];assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
old=json.load(open(P/'config.json'));(P/'precision-config.json').write_text(json.dumps(old,indent=2)+'\n');(P/'MEASURED.patch').write_text(old['runtime_patch']);p=R/'src/llama-context.cpp';s=p.read_text();(P/'llama-context.measured.cpp').write_text(s);needle='if (next_source == 24 && pool->op == GGML_OP_POOL_2D';assert s.count(needle)==1;s=s.replace(needle,'if (next_source == 24 && next_source < (int) model.hparams.n_layer() && pool->op == GGML_OP_POOL_2D');p.write_text(s)
with(P/'build-final-guard.log').open('wb')as f:subprocess.run(['cmake','--build','build-aliceai','--target','llama-server','-j','12'],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
new=old.copy();new['runtime_patch']=subprocess.check_output(['git','diff','--','src/llama-context.cpp','src/models/deepseek41.cpp'],cwd=R,text=True);new['source_library_hashes']={n:hashlib.file_digest((R/'build-aliceai/bin'/n).open('rb'),'sha256').hexdigest()for n in old['source_library_hashes']};(P/'config.json').write_text(json.dumps(new,indent=2)+'\n');(P/'FINAL.patch').write_text(new['runtime_patch']);(P/'llama-context.final.cpp').write_text(s)
# The only post-precision delta rejects an exhausted next-source search. This fixture has40layers/next24.
args=new['command'][1:];out=[];i=0
while i<len(args):
 k=args[i]
 if k in ['--host','--port','--alias']:i+=2;continue
 out.append(k)
 if k in ['--no-warmup','--no-mmproj-auto']:i+=1
 else:out.append(args[i+1]);i+=2
baseenv={k:v for k,v in os.environ.items()if not k.startswith(('GGML_','LLAMA_','CUDA_','DSV'))and k not in ['LD_PRELOAD','LD_LIBRARY_PATH']};baseenv.update(new['environment']);baseenv.update(LLAMA_DSV41_CANDIDATE_POOL_LOCAL='1',LD_PRELOAD=str(P/'metadata-trace.so'),DSV41_META_TRACE=str(P/'meta-trace-final-1.jsonl'))
with(P/'metadata-final-1.log').open('wb')as f:subprocess.run(['/tmp/dsv41-hc-on-long-indexer-20261103/metadata',str(P/'metadata-final-1.json'),'measure',*out],env=baseenv,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
a=json.load(open(P/'metadata-100352-1.json'));b=json.load(open(P/'metadata-final-1.json'));assert a['routes']==b['routes']and b['complete']
for x,y in zip(a['memory'],b['memory']):
 for k in ['main_model','main_context','main_compute','draft_and_taps_model','draft_and_taps_context','draft_and_taps_compute']:assert x[k]==y[k],k
rows=[json.loads(l)for l in(P/'meta-trace-final-1.jsonl').open()];assert {r['backend']for r in rows if r['name']=='candidate_block_score-20'and r['ne'][1]==512}=={'CUDA2'};assert {r['backend']for r in rows if r['name']=='candidate_pinned_block_score-20'and r['ne'][1]==512}=={'CUDA3'}
(P/'final-guard-summary.json').write_text(json.dumps({'complete':True,'delta':'Require next_source < n_layer before dev_layer(next_source); guard exhaustion sentinel24 on a24-layer future checkpoint. No changed path for current40-layer/next24 fixture.','precision_libraries':old['source_library_hashes'],'final_libraries':new['source_library_hashes'],'current_fixture_metadata_budgets_routes_equal':True,'final_pool':'CUDA2','final_selector':'CUDA3'},indent=2)+'\n');print('FINAL_GUARD_VALIDATED',flush=True)
