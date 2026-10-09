from pathlib import Path
import hashlib,json,math,re,subprocess
R=Path(__file__).resolve().parent;REPO=Path('/home/sergei/Github/llama.cpp')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text())
summary={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'production_source':'tools/server/server-context.cpp','source_sha256':sha(REPO/'tools/server/server-context.cpp'),'cases':{}}
compact=[]
for label,previous in [('preclear-native','native-ram-off'),('preclear-mtp','mtp-ram'),('preclear-dflash','dflash-ram')]:
 root=R/'server-runs'/label
 if not list(root.rglob('result.json')):continue
 d=next(root.rglob('result.json')).parent;result=load(d/'result.json');old=next((R.parent/'stage29/server-runs'/previous).rglob('result.json')).parent
 case={'process_status':result['status'],'response_count':len(list(d.glob('*-response.json'))),'ram_comparisons':[],'fresh_baseline_comparisons':[],'resident_comparisons':[],'parallel_responses':[]}
 for i in range(3):
  fresh=load(d/f'fresh1-{i}-response.json');oldfresh=load(old/f'fresh1-{i}-response.json')
  def equality(other):return {'tokens_equal':fresh['tokens']==other['tokens'],'text_equal':fresh['content']==other['content'],'available_probabilities_equal':fresh['completion_probabilities']==other['completion_probabilities']}
  case['fresh_baseline_comparisons'].append({'prompt':i,**equality(oldfresh)})
  for phase in ['reuse1','reuse2','fresh2','resident-fresh1','resident-reuse1','resident-reuse2','resident-fresh2']:
   f=d/f'{phase}-{i}-response.json'
   if not f.exists():continue
   cur=load(f);item={'prompt':i,'phase':phase,**equality(cur),'cache_n':cur['timings']['cache_n'],'prompt_n':cur['timings']['prompt_n'],'observed_probability_rows':sum(bool(x['top_logprobs']) for x in cur['completion_probabilities'])}
   case['resident_comparisons' if phase.startswith('resident') else 'ram_comparisons'].append(item)
  for wave in [1,2]:
   f=d/f'parallel{wave}-{i}-response.json'
   if f.exists():
    cur=load(f);case['parallel_responses'].append({'wave':wave,'prompt':i,'slot':cur.get('id_slot'),'token_count':len(cur['tokens']),'cache_n':cur['timings']['cache_n'],'prompt_n':cur['timings']['prompt_n']})
 for f in sorted(d.glob('*-response.json')):
  cur=load(f);compact.append({'case':label,'file':f.name,'slot':cur.get('id_slot'),'tokens':cur['tokens'],'content':cur['content'],'timings':cur['timings'],'probabilities': [{'index':i,**x} for i,x in enumerate(cur.get('completion_probabilities',[])) if x['top_logprobs']], 'source_sha256':sha(f)})
 comparisons=case['ram_comparisons']+case['resident_comparisons']+case['fresh_baseline_comparisons']
 case['all_comparisons_exact']=all(x['tokens_equal'] and x['text_equal'] and x['available_probabilities_equal'] for x in comparisons)
 case['ram_reuse_hits']=sum(x['cache_n']>0 for x in case['ram_comparisons'] if x['phase'].startswith('reuse'))
 case['ram_reuse_decodes_four_tokens']=all(x['prompt_n']==4 for x in case['ram_comparisons'] if x['phase'].startswith('reuse'))
 log=(d/'server.log').read_text();case['ram_restore_events']=log.count('found better prompt');case['checkpoint_restores']=log.count('restored context checkpoint')
 case['all_parallel_completed']=len(case['parallel_responses'])==6 and all(x['token_count']==64 for x in case['parallel_responses'])
 summary['cases'][label]=case
summary['total_responses']=sum(x['response_count'] for x in summary['cases'].values());summary['completed']=len(summary['cases'])==3 and all(x['process_status']=='passed' and x['all_comparisons_exact'] and x['ram_reuse_hits']==6 and x['ram_reuse_decodes_four_tokens'] and x['all_parallel_completed'] for x in summary['cases'].values())
summary['candidate_source_matches_repo']=sha(R/'preclear-server-context.cpp')==summary['source_sha256']==load(R/'preclear-build-manifest.json')['source_sha256']
summary['model_cuda_libraries_unchanged']=all(v==load(R/'preclear-build-manifest.json')['before_hashes'][n] for n,v in load(R/'preclear-build-manifest.json')['candidate_hashes'].items() if n!='libllama-server-impl.so')
(R/'final-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(R/'responses-compact.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='cases'});print({k:{n:v for n,v in x.items() if not isinstance(v,list)} for k,x in summary['cases'].items()})
