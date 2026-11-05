from pathlib import Path
import json,hashlib,shutil,subprocess
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts'
for name in ['reserve-probe.cpp','reserve-probe.so','reserve-probe-build.log','reserve-probe-build.json']:
 f=P/name;shutil.copy2(f,P/('initial-'+name))
(P/'metadata-0').rename(P/'initial-metadata-0')
p=P.parent/'.gitignore';p.write_text(p.read_text()+'artifacts/*/graphs.jsonl\n')
f=P/'reserve-probe.cpp';s=f.read_text().replace('#include <unordered_set>','#include <unordered_set>\n#include <set>\n#include <tuple>')
a=s.index('static void reserve_log(');b=s.index('extern "C" void ggml_backend_sched_reserve_size',a)
s=s[:a]+'''static void reserve_log(const char*kind,ggml_backend_sched_t s,ggml_cgraph*g) {
 const char*p=getenv("DSV_PREFILL_RESERVE_LOG");if(!p)return;
 int64_t nt=0;int deferred=0;bool target=false;int nn=ggml_graph_n_nodes(g);
 for(int i=0;i<nn;++i){auto*t=ggml_graph_node(g,i);if(!strcmp(t->name,"hc_attn_pre-0"))nt=t->ne[1];if(!strcmp(t->name,"hc_attn_pre-20"))target=true;if(!strncmp(t->name,"deferred_cache_update-",22))++deferred;}
 static std::set<std::tuple<int,int64_t,int>>seen;static size_t full=0;
 std::lock_guard<std::mutex>lock(log_mutex);
 json j={{"kind",kind},{"copies",ggml_backend_sched_get_n_copies(s)},{"node_count",nn},{"tokens",nt},{"target",target},{"deferred",deferred}};
 const auto key=std::make_tuple(nn,nt,deferred);
 if(target&&nt>6&&nt<=512&&nn<=12000&&full<12&&seen.insert(key).second){j["graph"]=graph_description(g);++full;}
 auto text=j.dump()+"\\n";FILE*f=fopen(p,"a");if(!f)abort();fwrite(text.data(),1,text.size(),f);fclose(f);
}
''' +s[b:];f.write_text(s)
cmd=json.loads((P/'reserve-probe-build.json').read_text())['command']
with(P/'reserve-probe-build.log').open('wb')as log:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
(P/'reserve-probe-build.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bounded':'Metadata only; at most12 complete target graphs with6<nt<=512 and<=12000nodes, unique nn/nt/deferred shape. Other calls store small headers. No added GPU events or eval callbacks.'},indent=2)+'\n')
raw=P/'initial-metadata-0/graphs.jsonl'
with raw.open('rb')as f:h=hashlib.file_digest(f,'sha256').hexdigest()
(P/'initial-metadata-cancellation.json').write_text(json.dumps({'complete':False,'reason':'Read-only reserve observer overcaptured repeated graphs; own metadata helper PID360654 deliberately SIGTERM after multi-GB trace. Initial raw trace retained locally and ignored. No runtime inference or quality result was claimed from this incomplete run.','raw_path':str(raw),'raw_bytes':raw.stat().st_size,'raw_sha256':h,'helper_exit':-15},indent=2)+'\n')
old=Path('/tmp/preflight-prefill-cache-deferral.py').read_text();tail=old[old.index("shutil.copy2(Old/'copies8/metadata',P/'metadata')"):];tail=tail.replace("shutil.copy2('/tmp/preflight-prefill-cache-deferral.py',P/'preflight.py')","shutil.copy2('/tmp/run-bounded-prefill-metadata.py',P/'preflight-bounded.py')")
script="from pathlib import Path\nimport json,subprocess,shutil,os,hashlib\nR=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';A=R/'docs/runtime-optimization/experiments/1017-kv-owner-audit-20261105/artifacts';Old=R/'docs/runtime-optimization/experiments/1017-prefill-residual-20261105/artifacts'\n"+tail
Path('/tmp/run-bounded-prefill-metadata.py').write_text(script);compile(script,'bounded','exec');shutil.copy2('/tmp/bound-prefill-metadata.py',P/'bound-metadata.py');print('Bounded metadata observer built; initial trace hashed/preserved.',flush=True)
