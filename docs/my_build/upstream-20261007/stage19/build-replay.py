from pathlib import Path
import json,subprocess,hashlib
R=Path(__file__).resolve().parent;old=R.parent/'stage18'
f=(R/'trace-fragment.cpp').read_text().replace('down_state->options->','down_state->precision->');(R/'trace-fragment.cpp').write_text(f)
s=(old/'candidate-replay.cpp').read_text();function=s[s.index('static void precision_replay'):s.index('int main(')]
function=function.replace('    auto cp = llama_context_default_params();','    const int row_count=forced.size();\n    auto cp = llama_context_default_params();')
function=function.replace('    cp.cb_eval = down_precision;\n    cp.cb_eval_user_data = &state;', '    TraceState trace{&state};\n    trace.enabled=capture;trace.prompt_tokens=prompt.size();\n    cp.cb_eval = trace_precision;\n    cp.cb_eval_user_data = &trace;')
function=function.replace('    options.label = label;', '    options.label = label;\n    if (capture) {\n        trace.trace=std::fopen((out+"/"+label+"-trace.jsonl").c_str(),"w");\n        check(trace.trace,"trace open failed");\n    }')
function=function.replace('start < 255','start < row_count-1').replace('255-start','row_count-1-start')
function=function.replace('        options.absolute_position = prompt.size()+start;', '''        options.absolute_position = prompt.size()+start;
        trace.first_input=start;
        bool focus=false;
        for (int col=0;col<batch.n_tokens;++col) for (int boundary : {256,512,768,1024}) {
            const int pos=prompt.size()+start+col;
            if (pos>=boundary-3 && pos<=boundary+2) focus=true;
        }
        if (capture && focus) {
            trace.directory=out+"/"+label+"-batch"+std::to_string(start);
            std::filesystem::create_directories(trace.directory);trace.event=0;
            trace.manifest=std::fopen((trace.directory+"/tensors.jsonl").c_str(),"w");
            check(trace.manifest,"capture manifest failed");
            std::ofstream alignment(trace.directory+"/alignment.json");
            alignment<<"{\\"batch_start\\":"<<start<<",\\"width\\":"<<batch.n_tokens<<",\\"prompt_tokens\\":"<<prompt.size()<<"}\\n";
        }''')
function=function.replace('        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);', '        if (trace.manifest) {std::fclose(trace.manifest);trace.manifest=nullptr;}\n        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);')
function=function.replace('    std::printf("DONE %s rows=256 vocab=%d prompt=%zu\\n", label.c_str(), vocab, prompt.size());', '    if (trace.trace) std::fclose(trace.trace);\n    std::printf("COMPRESSED_PADDING_CROPS %s %d\\n",label.c_str(),options.compressed_padding_crops);\n    std::printf("DONE %s rows=%d vocab=%d prompt=%zu\\n", label.c_str(), row_count, vocab, prompt.size());')
source='#include "llama-model.h"\n#include "legacy-callback.h"\n#include "general-callback.h"\nnamespace legacy {\nusing namespace ds19legacy;\n#include "trace-fragment.cpp"\n'+function+'}\nnamespace general {\nusing namespace ds14;\n#include "trace-fragment.cpp"\n'
function=function.replace('    const int row_count=forced.size();', '    options.compress_ratios.assign(model->hparams.dsv4_compress_ratios.begin(),model->hparams.dsv4_compress_ratios.begin()+model->hparams.n_layer());\n    const int row_count=forced.size();')
source+=function+'}\nusing namespace ds14;\n'
main=s[s.index('int main('):]
main=main.replace('forced(256)','forced(1024)').replace('input.gcount()>=256*4','input.gcount()>=1024*4')
a=main.index('        for (const std::string mode');b=main.index('        llama_model_free',a)
body='''        const std::string mode="decode-scalar-fa-upgate-hc-router-down-compressor";
        const std::string root=argv[4];
        for (const std::string dir : {"legacy", "legacy-no-trace", "general", "baseline"}) std::filesystem::create_directories(root+"/"+dir);
        for (int width : {1,3,4}) {
            ds19legacy::PrecisionOptions options{mode,false};
            legacy::precision_replay(model,prompt,forced,width,true,0,root+"/legacy",options);
        }
        {
            ds19legacy::PrecisionOptions options{mode,false};
            legacy::precision_replay(model,prompt,forced,4,false,0,root+"/legacy-no-trace",options);
        }
        for (int width : {1,2,3,4}) {
            PrecisionOptions options{mode,false};
            general::precision_replay(model,prompt,forced,width,true,0,root+"/general",options);
        }
        const std::string baseline_text=read_text("BASELINE_PROMPT");
        const int bn=-llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),nullptr,0,true,true);
        std::vector<llama_token> bp(bn),bf(95);
        check(llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),bp.data(),bn,true,true)==bn,"baseline tokenize failed");
        std::ifstream bi("BASELINE_PREFIX",std::ios::binary);
        check(bool(bi.read(reinterpret_cast<char *>(bf.data()),bf.size()*4)),"baseline prefix failed");
        for (int width : {1,4}) {
            PrecisionOptions options{mode,false};
            general::precision_replay(model,bp,bf,width,false,0,root+"/baseline",options);
        }
'''.replace('BASELINE_PROMPT',str(R.parent/'stage7/baseline-prompt.txt')).replace('BASELINE_PREFIX',str(R.parent/'stage9/forced-current-native.i32'))
source+=main[:a]+body+main[b:]
(R/'boundary-replay.cpp').write_text(source)
cmd=json.loads((R.parent/'stage17/explain-build-manifest.json').read_text())['command'];o=R.parent/'stage17'
cmd=[x.replace(str(o/'explain-replay.cpp'),str(R/'boundary-replay.cpp')).replace(str(o/'explain-replay'),str(R/'boundary-replay')) for x in cmd]
with (R/'build.txt').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,(R/'build.txt').read_text()
(R/'build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':0,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'boundary-replay.cpp',R/'legacy-callback.h',R/'general-callback.h',R/'trace-fragment.cpp']}},indent=2)+'\n')
s=(old/'run-v2.py').read_text().replace("R/'v2-replay'","R/'boundary-replay'").replace("R/'v2-output'","R/'boundary-output'").replace("old/'explain-native.i32'","R/'history.i32'").replace("R/'v2-run-manifest.json'","R/'run-manifest.json'").replace("R/'v2.log'","R/'replay.log'")
s=s.replace("'production_change':False", "'production_change':False,'rows':1024,'prefill_chunks':[19,4],'experiment':'Fresh append-only teacher-forced native response concatenation, fixed widths; no draft/rollback'")
(R/'run.py').write_text(s)
