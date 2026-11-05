from pathlib import Path
import os,time,json,subprocess,traceback
P=Path('/home/sergei/Github/llama.cpp/docs/runtime-optimization/experiments/1017-cpu-lookup-overlap-20261105/artifacts');pid=int((P/'native-controller.pid').read_text());j={'complete':False,'native_controller_pid':pid,'status':'waiting for clean native process to exit'}
def save():(P/'queue-results.json').write_text(json.dumps(j,indent=2)+'\n')
save()
try:
 while Path(f'/proc/{pid}').exists():time.sleep(5)
 n=json.loads((P/'native-results.json').read_text());assert n['complete']and n['all_outputs_counters_exact'],'native gate incomplete'
 delta={}
 for k in [8192,98304]:
  arms={a['label']:[r['pp']for r in a['requests']if r['prefix']==k and r['kind']=='measured']for a in n['arms']};mean=lambda v:sum(v)/len(v);off=(mean(arms['off-pre'])+mean(arms['off-post']))/2;delta[str(k)]=mean(arms['on'])/off-1
 j['native_PP_delta']=delta
 if max(delta.values())>.02 and min(delta.values())>-.02:
  j['status']='precision started after native complete';save()
  with(P/'precision-run.log').open('wb')as f:rc=subprocess.call(['python3',str(P/'precision.py')],stdout=f,stderr=subprocess.STDOUT,cwd=P)
  j['precision_exit']=rc;assert rc==0,'precision gate failed';j['status']='native and precision complete'
 else:j['status']='speed screen did not justify expanded quality gate; retain candidate OFF for review'
 j['complete']=True;save();print(j,flush=True)
except Exception as ex:j['error']=repr(ex);j['traceback']=traceback.format_exc();save();raise
