from pathlib import Path
import hashlib,json,subprocess
R=Path(__file__).resolve().parent
(R/'diagnostic-callback.h').write_bytes((R.parent/'stage15/diagnostic-callback.h').read_bytes())
s=(R.parent/'stage15/down-candidate.cpp').read_text()
fragment=(R.parent/'stage15/capture-fragment.cpp').read_text()
fragment=fragment.replace('PrecisionOptions * precision;','DownState * down_state;')
fragment=fragment.replace('controlled_precision(tensor, ask, state.precision)','down_precision(tensor, ask, state.down_state)')
(R/'capture-fragment.cpp').write_text(fragment)
s=s.replace('static void precision_replay(',fragment+'\nstatic void precision_replay(')
s=s.replace('    cp.cb_eval = down_precision;\n    cp.cb_eval_user_data = &state;','    CaptureState capture_state{&state};\n    cp.cb_eval = capture_precision;\n    cp.cb_eval_user_data = &capture_state;')
needle='        batch.n_tokens = std::min(start < switch_at ? 1 : width, 94-start);'
s=s.replace(needle,needle+'''
        capture_state.active=start==0 || (start<=18 && start+batch.n_tokens>18);
        if (capture_state.active) {
            const int target=start==0?0:18;
            capture_state.directory=out+"/"+label+"-input"+std::to_string(target);
            std::filesystem::create_directories(capture_state.directory);
            capture_state.manifest=std::fopen((capture_state.directory+"/tensors.jsonl").c_str(),"w");
            check(capture_state.manifest,"capture manifest failed");
            std::ofstream meta(capture_state.directory+"/alignment.json");
            meta<<"{\\\"input_index\\\":"<<target<<",\\\"output_index\\\":"<<target+1
                <<",\\\"batch_start\\\":"<<start<<",\\\"width\\\":"<<batch.n_tokens
                <<",\\\"column\\\":"<<target-start<<"}\\n";
            capture_state.event=0;
        }''')
s=s.replace('        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);','''        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);
        if (capture_state.active) {std::fclose(capture_state.manifest);capture_state.manifest=nullptr;}''')
(R/'capture-chain.cpp').write_text(s)
runner=(R.parent/'stage15/run-down-candidate.py').read_text().replace("'down-candidate'","'capture-chain'").replace("'down-candidate.cpp'","'capture-chain.cpp'").replace("'down-candidate-output'","'chain-capture-output'").replace("'down-candidate-manifest.json'","'chain-capture-manifest.json'").replace("'down-candidate.log'","'chain-capture.log'")
runner=runner.replace('scalar95 logits must match Stage14; measure whether scalar routed down/weight/sum resolves wide differences','all95 full logits at each width must match Stage15 down-candidate')
runner=runner.replace('after Stage14 intervention, scalarize routed down/weight/sum using live cached hidden, IDs and weights; no production change','capture all40 layer boundaries after Stage15 scalar down; inputs0/18, width4 input18 is column2 of batch16')
(R/'run-chain-capture.py').write_text(runner)
S=R.parent/'stage8/candidate-bin';repo=Path('/home/sergei/Github/llama.cpp')
cmd=['g++','-std=c++17','-O2',*['-I'+str(repo/p) for p in ['include','src','ggml/include']],str(R/'capture-chain.cpp'),'-L'+str(S),'-Wl,-rpath,'+str(S),'-lllama','-lggml','-lggml-base','-o',str(R/'capture-chain')]
x=subprocess.run(cmd,capture_output=True,text=True);(R/'build.txt').write_text(x.stdout+x.stderr)
(R/'build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':x.returncode,'sources':{n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['prepare-capture.py','diagnostic-callback.h','capture-fragment.cpp','capture-chain.cpp','run-chain-capture.py']}},indent=2)+'\n')
print(x.returncode,x.stderr)
raise SystemExit(x.returncode)
