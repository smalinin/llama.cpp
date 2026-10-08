from pathlib import Path
import hashlib,json,struct,subprocess

R=Path(__file__).resolve().parent
s=(R/'boundary-replay.cpp').read_text()
a=s.index('static void precision_replay',s.index('namespace general {'))
b=s.index('\n}\nusing namespace ds14;',a)
f=s[a:b]
f=f.replace('int switch_at = 0','int schedule = 0')
f=f.replace('(switch_at ? "-switch" + std::to_string(switch_at) : "")',
            '(schedule ? "-schedule" + std::to_string(schedule) : "") + "-rs" + std::to_string(rollback)')
f=f.replace('    for (int start = 0; start < row_count-1;) {',
'''    int attempt=0,removals=0,rejected=0;
    std::ofstream attempts(out+"/"+label+"-attempts.jsonl");
    for (int start = 0; start < row_count-1; ++attempt) {''')
f=f.replace('        batch.n_tokens = std::min(start < switch_at ? 1 : width, row_count-1-start);',
'''        const int append_widths[] = {1,4,2,3};
        const int rollback_widths[] = {4,3,2,4};
        const int requested = schedule == 1 ? append_widths[attempt%4] : schedule == 2 ? rollback_widths[attempt%4] : width;
        batch.n_tokens = std::min(requested,row_count-1-start);
        int keep=batch.n_tokens;
        if (schedule == 2) for (int boundary : {256,512,768,1024}) {
            const int pos=prompt.size()+start;
            if (pos>=boundary-5 && pos<=boundary+4) keep=std::min(keep,1+attempt%3);
        }''')
f=f.replace('        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);\n        start += batch.n_tokens;',
'''        for (int i = 0; i < keep; ++i) record(start+i+1, i);
        const int remove=batch.n_tokens-keep;
        if (remove) {
            check(rollback>=remove,"rollback exceeds snapshot depth");
            check(llama_memory_seq_rm(memory,0,prompt.size()+start+keep,-1),"partial rollback failed");
            ++removals;rejected+=remove;
        }
        attempts<<"{\\"start\\":"<<start<<",\\"position\\":"<<prompt.size()+start
            <<",\\"width\\":"<<batch.n_tokens<<",\\"keep\\":"<<keep<<",\\"removed\\":"<<remove<<"}\\n";
        start += keep;''')
f=f.replace('    std::printf("PADDING_CROPS',
            '    std::printf("ROLLBACKS %s %d events %d tokens\\n",label.c_str(),removals,rejected);\n    std::printf("PADDING_CROPS')
source='#include "llama-model.h"\n#include "general-callback.h"\nnamespace general {\nusing namespace ds14;\n#include "trace-fragment.cpp"\n'+f+'\n}\nusing namespace ds14;\n'
main=s[s.index('int main('):]
a=main.index('        for (const std::string dir');b=main.index('        llama_model_free',a)
body='''        for (const std::string dir : {"scalar-rs3","mixed-append","mixed-rollback","long-prompt"}) std::filesystem::create_directories(root+"/"+dir);
        for (const int which : {0,1,2}) {
            PrecisionOptions options{mode,false};
            const int width=which==0 ? 1 : 4,rollback=which==1 ? 0 : 3;
            const std::string dir=which==0 ? "scalar-rs3" : which==1 ? "mixed-append" : "mixed-rollback";
            general::precision_replay(model,prompt,forced,width,true,rollback,root+"/"+dir,options,which);
        }
        const std::string baseline_text=read_text("BASELINE_PROMPT");
        const int bn=-llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),nullptr,0,true,true);
        std::vector<llama_token> bp(bn),bf(512);
        check(llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),bp.data(),bn,true,true)==bn,"baseline tokenize failed");
        std::ifstream bi("LONG_PREFIX",std::ios::binary);
        check(bool(bi.read(reinterpret_cast<char *>(bf.data()),bf.size()*4)),"long prefix failed");
        for (int width : {1,4}) {
            PrecisionOptions options{mode,false};
            general::precision_replay(model,bp,bf,width,false,0,root+"/long-prompt",options);
        }
'''.replace('BASELINE_PROMPT',str(R.parent/'stage7/baseline-prompt.txt')).replace('LONG_PREFIX',str(R/'long-history.i32'))
source+=main[:a]+body+main[b:]
(R/'schedule-replay.cpp').write_text(source)
tokens=list(struct.unpack('<95i',(R.parent/'stage9/forced-current-native.i32').read_bytes()[:95*4]))
tokens+=list(struct.unpack('<1024i',(R/'history.i32').read_bytes()))[:417]
assert len(tokens)==512
(R/'long-history.i32').write_bytes(struct.pack('<512i',*tokens))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(R/'long-history-manifest.json').write_text(json.dumps({'kind':'Teacher-forced concatenation, not free generation or quality benchmark','tokens':512,'pieces':[{'path':str(R.parent/'stage9/forced-current-native.i32'),'tokens':95,'sha256':sha(R.parent/'stage9/forced-current-native.i32')},{'path':str(R/'history.i32'),'tokens':417,'sha256':sha(R/'history.i32')}],'sha256':sha(R/'long-history.i32')},indent=2)+'\n')
cmd=json.loads((R/'build-manifest.json').read_text())['command']
cmd=[x.replace(str(R/'boundary-replay'),str(R/'schedule-replay')) for x in cmd]
with (R/'schedule-build.txt').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,(R/'schedule-build.txt').read_text()
(R/'schedule-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':0,'sources':{p.name:sha(p) for p in [R/'schedule-replay.cpp',R/'general-callback.h',R/'trace-fragment.cpp']}},indent=2)+'\n')
s=(R/'run.py').read_text().replace("R/'boundary-replay'","R/'schedule-replay'").replace("R/'boundary-output'","R/'schedule-output'").replace("R/'run-manifest.json'","R/'schedule-run-manifest.json'").replace("R/'replay.log'","R/'schedule-replay.log'")
s=s.replace("'Fresh append-only teacher-forced native response concatenation, fixed widths; no draft/rollback'","'Teacher-forced fixed history: scalar rs3, mixed append, bounded partial rollback near padding boundaries, and longer prompt; no draft model'")
s=s.replace("'rows':1024", "'long_prefix_sha256':sha(R/'long-history.i32'),'rows':1024")
(R/'run-schedules.py').write_text(s)
