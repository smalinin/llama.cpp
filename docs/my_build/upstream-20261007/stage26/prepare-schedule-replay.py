#!/usr/bin/env python3
from pathlib import Path
import difflib,hashlib,json,subprocess

R=Path(__file__).resolve().parent
s=(R/'cache-replay.cpp').read_text().replace('#include <sstream>','#include <sstream>\n#include <iomanip>')
s=s.replace('static void replay(', '#include "schedule-fragment.h"\nstatic void replay(')
s=s.replace('const std::vector<llama_token> & warmup) {','const std::vector<llama_token> & warmup,const std::string & plan_file,const std::string & warm_plan,const std::string & reference_file) {')
needle='        for(int start=0;start<(int)warmup.size()-1;++start) {fill(warmup,start,1,prompt.size()+start,true);decode(state,batch,true);}'
assert s.count(needle)==1
s=s.replace(needle,'        if(!warm_plan.empty())warm_schedule(state,batch,warm_plan);else\n'+needle)
s=s.replace('    state.measured=true;', '    state.options.indexer_counts.clear();\n    state.measured=true;')
s=s.replace('    auto record=[&](int index,int row) {','''    int recorded_count=0,record_event=-1;bool native_prefix=true;
    std::ifstream scalar(reference_file,std::ios::binary);std::vector<float> scalar_values(vocab);
    rows<<std::setprecision(17);
    auto record=[&](int index,int row) {''')
s=s.replace('        output.write(reinterpret_cast<const char *>(values),vocab*4);int best=0;','''        const int p=(int)prompt.size()+index-1;
        const bool has_logits=plan_file.empty() || index==0 || (p>=7014 && p<=7024) || (p>=7395 && p<=7405);
        const long long offset=has_logits?(long long)output.tellp():-1;
        if(has_logits)output.write(reinterpret_cast<const char *>(values),vocab*4);
        bool exact=false;double max_abs=0,sum=0;
        if(scalar.is_open()) {
            scalar.seekg((long long)index*vocab*4);check(bool(scalar.read(reinterpret_cast<char *>(scalar_values.data()),vocab*4)),"scalar row read failed");
            exact=std::memcmp(values,scalar_values.data(),vocab*4)==0;
            if(!exact && native_prefix)for(int i=0;i<vocab;++i) {const double d=double(values[i])-scalar_values[i];max_abs=std::max(max_abs,std::abs(d));sum+=d*d;}
        }
        ++recorded_count;
        int best=0;''')
s=s.replace('        rows<<"{\\"argmax_no_eog\\":', '''        rows<<"{\\"event\\":"<<record_event<<",\\"pos\\":"<<p<<",\\"native_prefix\\":"<<(native_prefix?"true":"false")
            <<",\\"has_logits\\":"<<(has_logits?"true":"false")<<",\\"logits_offset\\":"<<offset<<",\\"scalar_available\\":"<<(scalar.is_open()?"true":"false")
            <<",\\"scalar_exact\\":"<<(exact?"true":"false")<<",\\"max_abs\\":"<<max_abs<<",\\"rms\\":"<<std::sqrt(sum/vocab)<<",\\"argmax_no_eog\\":''')
needle='    for(int start=0;start<(int)forced.size()-1;) {'
assert s.count(needle)==1
s=s.replace(needle,'''    if(!plan_file.empty()) {
        for(const auto & e:read_schedule(plan_file)) {
            check(llama_memory_seq_rm(memory,0,e.pos,-1),"verification rollback failed");
            fill_schedule(batch,e);decode(state,batch,true);check_schedule_layout(state,e);
            native_prefix=true;record_event=e.event;
            for(int col=0;col<e.width;++col) {
                const int input=e.pos-(int)prompt.size()+col;
                check(input>=0 && input+1<(int)forced.size(),"verification history bounds failed");
                native_prefix=native_prefix && e.tokens[col]==forced[input];record(input+1,col);
            }
        }
    } else
'''+needle)
s=s.replace('<<forced.size()<<",\\"vocab\\"', '<<recorded_count<<",\\"vocab\\"')
s=s.replace('directory.c_str(),forced.size(),width,cached','directory.c_str(),size_t(recorded_count),width,cached')
s=s.replace('std::string label,prompt,prefix,warm;','std::string label,prompt,prefix,warm,plan,warm_plan,reference;')
s=s.replace('row>>warm;replay(', 'row>>warm;row>>plan>>warm_plan>>reference;if(plan=="-")plan.clear();if(warm_plan=="-")warm_plan.clear();if(reference=="-")reference.clear();replay(')
s=s.replace('tokens(warm.empty()?prefix:warm,model));','tokens(warm.empty()?prefix:warm,model),plan,warm_plan,reference);')
assert 'recorded_count<<' in s
(R/'verify-replay.cpp').write_text(s)
(R/'verify-replay.patch').write_text(''.join(difflib.unified_diff((R/'cache-replay.cpp').read_text().splitlines(True),s.splitlines(True),fromfile='stage26/cache-replay.cpp',tofile='stage26/verify-replay.cpp')))
cmd=[p.replace('/cache-replay','/verify-replay') for p in json.loads((R/'replay-build-manifest.json').read_text())['command']]
result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(R/'verify-build.log').write_text(result.stdout)
sources={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['verify-replay.cpp','schedule-fragment.h','general-callback.h','raw-layout.h']}
(R/'verify-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':result.returncode,'sources':sources},indent=2)+'\n')
print(result.stdout);raise SystemExit(result.returncode)
