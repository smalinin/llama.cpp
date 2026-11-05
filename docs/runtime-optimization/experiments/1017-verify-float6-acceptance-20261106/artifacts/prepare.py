from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path('/home/sergei/Github/llama.cpp');B=R/'docs/runtime-optimization/experiments/1017-verify-float6-20261106/artifacts';F=R/'docs/runtime-optimization/experiments/1017-full-precision-forensic-20261105/artifacts';M=R/'docs/runtime-optimization/experiments/1017-verify-moe5plus1-20261105/artifacts';W=R/'docs/runtime-optimization/experiments/1017-verify-width-screen-20261105/artifacts';P=R/'docs/runtime-optimization/experiments/1017-verify-float6-acceptance-20261106/artifacts';P.mkdir(parents=True)
(P.parent/'.gitignore').write_text('artifacts/*-harness\nartifacts/*.so\nartifacts/__pycache__/\nartifacts/bundle/\n!*.log\n')
sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
cfg=json.loads((B/'config.json').read_text());assert all(sha(f)==h for f,h in cfg['source_library_hashes'].items());assert all(sha(R/f)==h for f,h in cfg['source_files'].items());cfg['head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();(P/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
for root,name in [(B,'memory_monitor.py'),(F,'inputs.json'),(F,'fault.cpp'),(W,'counts.cpp'),(W,'counts.h')]:shutil.copy2(root/name,P/name)
shutil.copy2(B/'build.json',P/'runtime-build.json');shutil.copy2(B/'admission.json',P/'prior-admission.json')
s=(M/'precision.cpp').read_text();s=s.replace('#include <cmath>','#include <cmath>\n#include <dlfcn.h>\n#include "counts.h"')
anchor='static Floats apply('
pos=s.index(anchor)
helpers='''static json dispatch_records=json::array();
static std::string current_corpus;
using CountBegin=void(*)();
using CountEnd=void(*)(uint64_t*);
static void count_begin(){static auto f=reinterpret_cast<CountBegin>(dlsym(RTLD_DEFAULT,"dsv_width_count_begin"));need(f,"counter begin");f();}
static void count_end(int pos){static auto f=reinterpret_cast<CountEnd>(dlsym(RTLD_DEFAULT,"dsv_width_count_end"));need(f,"counter end");uint64_t v[counter_count];f(v);json counts;for(int i=0;i<counter_count;++i)counts[counter_names[i]]=v[i];dispatch_records.push_back({{"corpus",current_corpus},{"position",pos},{"tokens",6},{"counts",counts}});}
'''
s=s[:pos]+helpers+s[pos:]
old='need(llama_decode(t,b)==0,"target decode");Floats v;if(out)v=logits(t,all?ids.size():1);need(common_speculative_process(s,b),"feature process");';new='bool count=all&&out&&ids.size()==6;if(count)count_begin();need(llama_decode(t,b)==0,"target decode");Floats v;if(out)v=logits(t,all?ids.size():1);if(count)count_end(pos);need(common_speculative_process(s,b),"feature process");';assert s.count(old)==1;s=s.replace(old,new)
s=s.replace('auto save=[&]{std::ofstream(dest/"results.json")','auto save=[&]{out["dispatch"]=dispatch_records;std::ofstream(dest/"results.json")')
s=s.replace('std::string id=corpus["id"];auto tok=','std::string id=corpus["id"];current_corpus=id;auto tok=')
# Append a six-query NLL cohort after the original fault/retry sequence.
line=next(x for x in s.splitlines() if x.startswith('restore(t,bs);restore(d,ds);json rows=json::array();double mean=0;'))
wide=line.replace('restore(t,bs);','{restore(t,bs);',1).replace('off<128;off+=4','off<126;off+=6').replace('90112+off,4','90112+off,6').replace('i<4','i<6').replace('"nll128"','"nll126-wide6"').replace('mean/128','mean/126')+'}\n'
needle='}}out["complete"]=true;';assert s.count(needle)==1;s=s.replace(needle,wide+needle)
(P/'lifecycle.cpp').write_text(s)
cmd=json.loads((F/'build.json').read_text())['command'];cmd=[str(P/'lifecycle.cpp')if x==str(F/'full-forensic.cpp')else str(P/'lifecycle-harness')if x==str(F/'full-forensic-harness')else x for x in cmd]
commands=[cmd,['/usr/bin/c++','-O2','-std=c++17','-shared','-fPIC','-I'+str(R/'ggml/include'),'-I/usr/local/cuda/include',str(P/'counts.cpp'),'-ldl','-o',str(P/'counts.so')],['/usr/bin/c++','-O2','-std=c++17','-shared','-fPIC','-I'+str(R/'ggml/include'),str(P/'fault.cpp'),'-ldl','-o',str(P/'fault.so')]]
with(P/'build.log').open('wb')as log:
 for cmd in commands:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
(P/'build.json').write_text(json.dumps(dict(complete=True,commands=commands,source_sha256=sha(P/'lifecycle.cpp'),harness_sha256=sha(P/'lifecycle-harness'),counter_sha256=sha(P/'counts.so'),fault_sha256=sha(P/'fault.so'),inputs_sha256=sha(P/'inputs.json'),scope='Existing original precision sequence plus existing lightweight counters at six-query target decode and added NLL126/6 after original failure/retry. No extra GPU reads for counters, no callback/CUPTI, no runtime/source/default change. Counter timings are diagnostic only.'),indent=2)+'\n')
print('LIFECYCLE_BUILT')
