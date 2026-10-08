from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent;old=R.parent/'stage19'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(R/'general-callback.h').write_bytes((old/'general-callback.h').read_bytes())
boundaries='{3033,3072,3289,3328,6617,6657}'
s=(old/'trace-fragment.cpp').read_text().replace('{256,512,768,1024}',boundaries)
(R/'trace-fragment.cpp').write_text(s)
s=(old/'boundary-replay.cpp').read_text()
a=s.index('static void precision_replay',s.index('namespace general {'));b=s.index('\n}\nusing namespace ds14;',a)
f=s[a:b].replace('{256,512,768,1024}',boundaries)
f=f.replace('    for (int prompt_end : {(int) prompt.size()-516, (int) prompt.size()-4, (int) prompt.size()}) {',
'''    std::vector<int> prompt_ends;
    for (int end=2048;end<(int)prompt.size()-516;end+=2048) prompt_ends.push_back(end);
    for (int end : {(int)prompt.size()-516,(int)prompt.size()-4,(int)prompt.size()}) if (end>0) prompt_ends.push_back(end);
    std::ofstream prefill(out+"/prefill-chunks.json");
    prefill<<"[";
    bool first_chunk=true;
    for (int prompt_end : prompt_ends) {''')
f=f.replace('        batch.n_tokens = prompt_end-prompt_start;',
'''        batch.n_tokens = prompt_end-prompt_start;
        check(batch.n_tokens<=2048,"prefill batch overflow");
        prefill<<(first_chunk ? "" : ",")<<batch.n_tokens;first_chunk=false;''')
f=f.replace('    const int vocab = llama_vocab_n_tokens', '    prefill<<"]\\n";prefill.close();\n    const int vocab = llama_vocab_n_tokens')
source='#include "llama-model.h"\n#include "general-callback.h"\nnamespace general {\nusing namespace ds14;\n#include "trace-fragment.cpp"\n'+f+'\n}\nusing namespace ds14;\n'
main=s[s.index('int main('):]
a=main.index('        const std::string mode=');b=main.index('        llama_model_free',a)
body='''        const std::string mode="decode-scalar-fa-upgate-hc-router-down-compressor";
        const std::string root=argv[4];
        std::ofstream metadata(root+"/model-hparams.json");
        metadata<<"{\\"swa\\":"<<model->hparams.n_swa<<",\\"indexer_top_k\\":"<<model->hparams.indexer_top_k<<",\\"compress_ratios\\":[";
        for (uint32_t layer=0;layer<model->hparams.n_layer();++layer) metadata<<(layer?",":"")<<model->hparams.dsv4_compress_ratios[layer];
        metadata<<"]}\\n";metadata.close();
        const auto history=forced;
        for (int which : {0,1,2}) {
            const int lengths[]={3033,3289,6617};
            const std::string names[]={"k4096","ratio1","ratio2"};
            auto p=prompt;
            for (int i=0;(int)p.size()<lengths[which];++i) p.push_back(history[i%history.size()]);
            forced.assign(history.begin(),history.begin()+96);
            const std::string out=root+"/"+names[which];std::filesystem::create_directories(out);
            std::ofstream(out+"/prompt.i32",std::ios::binary).write(reinterpret_cast<const char *>(p.data()),p.size()*4);
            std::ofstream(out+"/forced.i32",std::ios::binary).write(reinterpret_cast<const char *>(forced.data()),forced.size()*4);
            const std::vector<int> widths=which==0 ? std::vector<int>{1,4} : std::vector<int>{1,2,3,4};
            for (int width : widths) {
                PrecisionOptions options{mode,false};
                general::precision_replay(model,p,forced,width,true,0,out,options);
            }
            if (which==2) {
                const std::string control=root+"/ratio2-no-trace";std::filesystem::create_directories(control);
                PrecisionOptions options{mode,false};
                general::precision_replay(model,p,forced,4,false,0,control,options);
            }
        }
        const std::string baseline_text=read_text("BASELINE_PROMPT");
        const int bn=-llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),nullptr,0,true,true);
        std::vector<llama_token> bp(bn),bf(95);
        check(llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),bp.data(),bn,true,true)==bn,"baseline tokenize failed");
        std::ifstream bi("BASELINE_PREFIX",std::ios::binary);
        check(bool(bi.read(reinterpret_cast<char *>(bf.data()),bf.size()*4)),"baseline prefix failed");
        std::filesystem::create_directories(root+"/baseline");
        for (int width : {1,4}) {
            PrecisionOptions options{mode,false};
            general::precision_replay(model,bp,bf,width,false,0,root+"/baseline",options);
        }
'''.replace('BASELINE_PROMPT',str(R.parent/'stage7/baseline-prompt.txt')).replace('BASELINE_PREFIX',str(R.parent/'stage9/forced-current-native.i32'))
source+=main[:a]+body+main[b:]
(R/'sparse-replay.cpp').write_text(source)
cmd=json.loads((old/'build-manifest.json').read_text())['command']
cmd=[x.replace(str(old/'boundary-replay'),str(R/'sparse-replay')) for x in cmd]
with (R/'build.txt').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,(R/'build.txt').read_text()
(R/'build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':0,'sources':{p.name:sha(p) for p in [R/'sparse-replay.cpp',R/'general-callback.h',R/'trace-fragment.cpp']},'reused_header_sha256':sha(old/'general-callback.h')},indent=2)+'\n')
s=(old/'run.py').read_text().replace("R/'boundary-replay'","R/'sparse-replay'").replace("R/'boundary-output'","R/'model-output'").replace("R/'history.i32'","R.parent/'stage19/history.i32'")
s=s.replace("'rows':1024,'prefill_chunks':[19,4]", "'rows_per_window':96,'prompt_lengths':[3033,3289,6617]")
s=s.replace('Fresh append-only teacher-forced native response concatenation, fixed widths; no draft/rollback','Fixed-history windows across K4096 and actual ratio1/ratio2 sparse thresholds; no draft/rollback')
(R/'run.py').write_text(s)
