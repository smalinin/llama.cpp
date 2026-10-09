#!/usr/bin/env python3
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess

R=Path(__file__).resolve().parent
OLD=R.parent/'stage25'
SNAP=R.parent/'stage8/candidate-bin'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

s=(OLD/'server-context-general.cpp').read_text()
s=s.replace('struct server_context_impl {\n','''struct server_context_impl {
    std::ofstream ds26_capture;
    std::string ds26_directory;
    std::vector<uint8_t> ds26_eog;
    int ds26_event = 0;
''')
needle='        vocab = llama_model_get_vocab(model_tgt);'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
        if (const char * directory = std::getenv("LLAMA_DSV41_CAPTURE")) {
            ds26_directory = directory;
            std::filesystem::create_directories(ds26_directory);
            ds26_capture.open(ds26_directory + "/events.jsonl");
            GGML_ASSERT(ds26_capture.good());
            ds26_eog.resize(llama_vocab_n_tokens(vocab));
            for (int i=0; i<(int)ds26_eog.size(); ++i) ds26_eog[i]=llama_vocab_is_eog(vocab,i);
        }
''')
needle='            ds14_options.decoding = false;'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
            if (ds26_capture.is_open() && ds14_generating && ret==0) {
                const int event=ds26_event++;
                ds26_capture << "{\\"event\\":" << event << ",\\"pos\\":" << batch_view.pos[0]
                             << ",\\"width\\":" << batch_view.n_tokens << ",\\"tokens\\":[";
                for (int i=0; i<batch_view.n_tokens; ++i) ds26_capture << (i?",":"") << batch_view.token[i];
                ds26_capture << "],\\"rows\\":[";
                for (int col=0; col<batch_view.n_tokens; ++col) {
                    const float * values=llama_get_logits_ith(ctx_tgt,col);
                    GGML_ASSERT(values);
                    int best=0, non_eog=-1;
                    for (int i=0; i<(int)ds26_eog.size(); ++i) {
                        if (values[i]>values[best]) best=i;
                        if (!ds26_eog[i] && (non_eog<0 || values[i]>values[non_eog])) non_eog=i;
                    }
                    const int p=batch_view.pos[col];
                    std::string file;
                    if ((p>=7014 && p<=7024) || (p>=7395 && p<=7405)) {
                        file="event-"+std::to_string(event)+"-col-"+std::to_string(col)+".f32";
                        std::ofstream(ds26_directory+"/"+file,std::ios::binary)
                            .write(reinterpret_cast<const char *>(values),ds26_eog.size()*4);
                    }
                    ds26_capture << (col?",":"") << "{\\"pos\\":" << p << ",\\"argmax\\":" << best
                                 << ",\\"argmax_no_eog\\":" << non_eog << ",\\"file\\":\\"" << file << "\\"}";
                }
                ds26_capture << "]}\\n";ds26_capture.flush();
            }
''')
(R/'server-context-capture.cpp').write_text(s)
(R/'server-capture.patch').write_text(''.join(difflib.unified_diff((OLD/'server-context-general.cpp').read_text().splitlines(True),s.splitlines(True),fromfile='stage25/server-context-general.cpp',tofile='stage26/server-context-capture.cpp')))
(R/'server-build').mkdir(exist_ok=True)
OUT=R/'experiment-bin'
assert not OUT.exists();shutil.copytree(SNAP,OUT,symlinks=True)
old=json.loads((OLD/'build-manifest.json').read_text())
commands=[[p.replace('/stage25/','/stage26/').replace('/server-context-general.cpp','/server-context-capture.cpp') for p in command] for command in old['commands']]
archive=next(Path(p) for p in commands[1] if p.endswith('.a'))
shutil.copy2('/home/sergei/Github/llama.cpp/build-glm53/tools/server/libserver-context.a',archive)
meta={k:v for k,v in old.items() if k not in ['commands','sources','experiment_hashes','exit_code']}
meta.update(commands=commands,head=subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip(),sources={n:sha(R/n) for n in ['general-callback.h','raw-layout.h','server-context-capture.cpp','server-capture.patch']})
with (R/'server-build.log').open('w') as log:
    for cmd in commands:
        result=subprocess.run(cmd,cwd=old['link_cwd'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(result.returncode)
libs=json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())
meta.update(exit_code=0,experiment_hashes={n:sha(OUT/n) for n in libs})
assert [n for n,h in libs.items() if meta['experiment_hashes'][n]!=h]==['libllama-server-impl.so']
(R/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Capture server compiled; callback and numerical model code unchanged')
