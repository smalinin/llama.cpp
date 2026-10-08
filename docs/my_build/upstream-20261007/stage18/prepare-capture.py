from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent
old=R.parent/'stage17'
shutil.copy2(old/'diagnostic-callback.h',R/'diagnostic-callback.h')
f=(R.parent/'stage16/capture-fragment.cpp').read_text()
a=f.index('        const int layer = std::stoi')
b=f.index('\n    }\n    return true;',a)
f=f[:a]+f[b:]
f=f.replace('    if (ask) return requested || boundary;', '''    const bool fa = state.active && tensor->op == GGML_OP_FLASH_ATTN_EXT &&
        (std::string(tensor->src[0]->name).find("q-0 ") == 0 ||
         std::string(tensor->src[0]->name).find("q-1 ") == 0 ||
         std::string(tensor->src[0]->name).find("q-2 ") == 0);
    if (ask) return requested || boundary || fa;''')
f=f.replace('    return true;\n}', '''    if (fa) {
        const std::string owner = "FA-" + std::string(tensor->src[0]->name).substr(2,1);
        save_boundary(tensor, state, owner, "output");
        for (int i = 0; i < 5; ++i) if (tensor->src[i]) save_boundary(tensor->src[i], state, owner, "src"+std::to_string(i));
        std::ofstream params(state.directory+"/"+owner+"-params.json");
        params << "{\\"op_params_u32\\":[";
        for (size_t i = 0; i < sizeof(tensor->op_params)/4; ++i) {
            uint32_t word;
            std::memcpy(&word,tensor->op_params+i,4);
            params << (i ? "," : "") << word;
        }
        params << "]}\\n";
    }
    return true;
}''')
(R/'capture-fragment.cpp').write_text(f)
s=(old/'explain-replay.cpp').read_text().replace('using namespace ds14;', 'using namespace ds14;\n#include "capture-fragment.cpp"')
s=s.replace('    cp.cb_eval = down_precision;\n    cp.cb_eval_user_data = &state;', '    CaptureState cap{&state};\n    cp.cb_eval = capture_precision;\n    cp.cb_eval_user_data = &cap;')
s=s.replace('        const int target=start==0?0:86;','''        cap.active = (start <= 231 && 231 < start+batch.n_tokens) || (start <= 232 && 232 < start+batch.n_tokens);
        if (cap.active) {
            cap.directory = out+"/"+label+"-batch"+std::to_string(start);
            std::filesystem::create_directories(cap.directory);
            cap.manifest = std::fopen((cap.directory+"/tensors.jsonl").c_str(),"w");
            check(cap.manifest,"manifest failed");
            cap.event = 0;
            std::ofstream align(cap.directory+"/alignment.json");
            align << "{\\"batch_start\\":" << start << ",\\"width\\":" << batch.n_tokens << ",\\"prompt_tokens\\":" << prompt.size() << "}\\n";
        }''')
s=s.replace('        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);','''        if (cap.active) { std::fclose(cap.manifest); cap.manifest = nullptr; cap.active = false; }
        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);''')
s=s.replace(', "decode-scalar-fa-upgate-hc-router-down-float2d"','')
(R/'capture-replay.cpp').write_text(s)
cmd=json.loads((old/'explain-build-manifest.json').read_text())['command']
cmd=[x.replace(str(old/'explain-replay.cpp'),str(R/'capture-replay.cpp')).replace(str(old/'explain-replay'),str(R/'capture-replay')) for x in cmd]
with (R/'capture-build.txt').open('w') as log:
 result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(R/'capture-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':result.returncode,'source_sha256':sha(R/'capture-replay.cpp'),'header_sha256':sha(R/'diagnostic-callback.h'),'fragment_sha256':sha(R/'capture-fragment.cpp')},indent=2)+'\n')
assert result.returncode==0,(R/'capture-build.txt').read_text()
