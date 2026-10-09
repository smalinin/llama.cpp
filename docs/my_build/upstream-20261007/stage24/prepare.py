#!/usr/bin/env python3
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;OLD=R.parent/'stage23';S=R.parent/'stage8/candidate-bin'
mode='decode-scalar-fa-upgate-hc-router-down-compressor-indexer'
s=(OLD/'general-callback.h').read_text()
s=s.replace('    std::map<std::string,int> matmul_counts;','    std::map<std::string,int> matmul_counts;\n    std::map<std::string,int> indexer_counts;')
s=s.replace('static bool selected_float_matmul(','''static bool indexer_projection(const ggml_tensor * tensor) {
    const std::string name=tensor->src[0]->name;
    const std::string suffix=".indexer.proj.weight";
    return name.find("blk.")==0 && name.size()>suffix.size() && name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
}

static bool selected_float_matmul(''')
s=s.replace('return options.mode.find("float2d") != std::string::npos ||','return options.mode.find("float2d") != std::string::npos ||\n        (options.mode.find("indexer") != std::string::npos && indexer_projection(tensor)) ||')
s=s.replace('    options.matmul_counts[tensor->name] += width;','    options.matmul_counts[tensor->name] += width;\n    if (indexer_projection(tensor)) options.indexer_counts[tensor->src[0]->name] += width;')
assert s!=(OLD/'general-callback.h').read_text();(R/'general-callback.h').write_text(s)
(R/'general-callback.patch').write_text(''.join(difflib.unified_diff((OLD/'general-callback.h').read_text().splitlines(True),s.splitlines(True),fromfile='stage23/general-callback.h',tofile='stage24/general-callback.h')))
shutil.copy2(OLD/'raw-layout.h',R/'raw-layout.h');shutil.copy2(OLD/'run-replay.py',R/'run-replay.py')
s=(R.parent/'stage22/cache-replay.cpp').read_text().replace('decode-scalar-fa-upgate-hc-router-down-compressor',mode)
s=s.replace('    llama_batch_free(batch);llama_free(state.ctx);','''    std::ofstream counts(directory+"/indexer-counts.json");counts<<"{";bool first=true;
    for(const auto & item:state.options.indexer_counts) {counts<<(first?"":",")<<"\\""<<item.first<<"\\":"<<item.second;first=false;}
    counts<<"}\\n";counts.close();
    check(state.options.indexer_counts.empty()==(width==1),"indexer selection count failed");
    llama_batch_free(batch);llama_free(state.ctx);''')
(R/'cache-replay.cpp').write_text(s)
lines=(R.parent/'stage22/cases-all.txt').read_text().splitlines();cases=[l for l in lines if l.split()[0] in ['baseline-w1','baseline-w4','sparse-control-w4','history-control-w4']]
prompt=(R.parent/'stage22/inputs/ratio2.i32').resolve();history=(R.parent/'stage19/history.i32').resolve()
for kind in ['cache','fresh']:
 for width in [1,4,2,3]:cases.append(f'{kind}-history-w{width} {prompt} {history} {width} {int(kind=="cache")}')
(R/'cases.txt').write_text('\n'.join(cases)+'\n')
s=(R.parent/'stage13/isolated-replay.cpp').read_text().replace('#include "ggml.h"','#include "general-callback.h"\n#include "ggml.h"')
s=s.replace('w = ggml_new_tensor_2d(wc,type,n,m);','w = ggml_new_tensor_2d(wc,type,n,m);\n        ggml_set_name(w,"blk.24.indexer.proj.weight");')
s=s.replace('            auto graph = ggml_new_graph(ctx);','''            ds14::PrecisionOptions options{"'''+mode+'''",false};options.decoding=true;
            if(mode=="projection")ds14::controlled_precision(y,true,&options);
            auto graph = ggml_new_graph(ctx);''')
s=s.replace('            std::vector<float> result(size_t(m)*width), repeat(result.size());','            if(mode=="projection")ds14::controlled_precision(y,false,&options);\n            std::vector<float> result(size_t(m)*width), repeat(result.size());')
s=s.replace('            ggml_backend_tensor_get(y,repeat.data(),0,repeat.size()*4);','            if(mode=="projection")ds14::controlled_precision(y,false,&options);\n            ggml_backend_tensor_get(y,repeat.data(),0,repeat.size()*4);')
(R/'projection-check.cpp').write_text(s)
for old,new in [(R.parent/'stage22/cache-replay.cpp',R/'cache-replay.cpp'),(R.parent/'stage13/isolated-replay.cpp',R/'projection-check.cpp')]:
 (R/(new.stem+'.patch')).write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),new.read_text().splitlines(True),fromfile=str(old.relative_to(R.parent)),tofile='stage24/'+new.name)))
base=json.loads((OLD/'replay-build-manifest.json').read_text())['command'];base=[x.replace('/stage23/','/stage24/') for x in base]
for name,cmd in [('replay',base),('projection',[x.replace('/cache-replay','/projection-check') for x in base])]:
 if name=='projection':cmd.insert(cmd.index('-lllama'),'-lggml-cuda')
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(R/f'{name}-build.log').write_text(p.stdout)
 (R/f'{name}-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':p.returncode,'sources':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in (['cache-replay.cpp','general-callback.h','raw-layout.h'] if name=='replay' else ['projection-check.cpp','general-callback.h'])}},indent=2)+'\n')
 print(name,'build',p.returncode,p.stdout);assert p.returncode==0
shutil.copy2(OLD/'source-selection.json',R/'source-selection.json');p=R/'source-selection.json';v=json.loads(p.read_text());v['head']=subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip();p.write_text(json.dumps(v,indent=2)+'\n')
