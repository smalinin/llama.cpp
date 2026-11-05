from pathlib import Path
import json,collections,subprocess,sys

def main():
 p=Path(sys.argv[1]);out=Path(sys.argv[2]);records=[];locations=collections.defaultdict(set)
 for line in p.read_text().splitlines():
  j=json.loads(line)
  if j.get('event')!='stack':continue
  records.append(j)
  for fr in j['frames']:
   f=fr['file']
   if any(x in f for x in ['maya-ds4','libggml','libllama']) and Path(f).exists():locations[f].add(fr['offset'])
 symbols={}
 for f,offsets in locations.items():
  ns=sorted(offsets);r=subprocess.run(['addr2line','-f','-C','-e',f],input=''.join(hex(x)+'\n'for x in ns),text=True,capture_output=True,check=True);lines=r.stdout.splitlines();assert len(lines)==2*len(ns)
  for i,n in enumerate(ns):symbols[(f,n)]={'function':lines[2*i],'source':lines[2*i+1]}
 summary=collections.Counter();samples=[];funcs=collections.Counter()
 for j in records:
  names=[symbols[(f['file'],f['offset'])]['function']for f in j['frames']if(f['file'],f['offset'])in symbols]
  if any('::draft_('in n or'Model::draft('in n for n in names):role='draft'
  elif any('::verify('in n for n in names):role='verify'
  elif any('flow_'in n for n in names):role='prefill_flow'
  elif any('::step('in n for n in names):role='target_step'
  else:role='worker_or_other'
  summary[(j['capture'],j['phase'],j['api'],role)]+=1
  for n in set(names):funcs[(j['phase'],n)]+=1
  if len([x for x in samples if x['role']==role and x['phase']==j['phase']])<3:samples.append({'role':role,'phase':j['phase'],'api':j['api'],'functions':names})
 result={'records':len(records),'symbols':[{'file':f,'offset':n,**v}for(f,n),v in symbols.items()],'samples_by_role':[{'capture':k[0],'phase':k[1],'api':k[2],'role':k[3],'samples':v}for k,v in summary.items()],'top_functions':[{'phase':k[0],'function':k[1],'samples':v}for k,v in funcs.most_common(45)],'sample_stacks':samples,'scope':'Periodic per-thread API callstack samples. Counts are not runtime call totals or time shares, and not statistical CPU IP sampling. Async worker stacks may omit parent draft/verify scope.'};out.write_text(json.dumps(result,indent=2)+'\n');print('Symbolized',len(records),'samples and',len(symbols),'locations')
 for x in samples:print(x['phase'],x['role'],x['api'],x['functions'][:3])
if __name__=='__main__':main()
