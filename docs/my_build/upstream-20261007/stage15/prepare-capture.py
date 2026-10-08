from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parent
(R/'diagnostic-callback.h').write_bytes((R.parent/'stage14/diagnostic-callback.h').read_bytes())
s=(R.parent/'stage14/extended-replay.cpp').read_text()
fragment=r'''
struct CaptureState {
    PrecisionOptions * precision;
    bool active = false;
    std::string directory;
    FILE * manifest = nullptr;
    int event = 0;
};
static bool capture_boundary(const std::string & name) {
    for (int layer = 0; layer < 40; ++layer) {
        for (const std::string prefix : {"hc_mixes", "hc_pre", "hc_post", "hc_comb", "hc_attn_pre", "attn_norm", "attn_out", "hc_attn_post", "hc_ffn_pre", "ffn_norm", "ffn_moe_logits", "ffn_moe_topk", "ffn_moe_weights_scaled", "ffn_moe_swiglu_limited", "ffn_moe_out", "ffn_out", "l_last"}) {
            if (name == prefix+"-"+std::to_string(layer)) return true;
        }
    }
    return false;
}
static void save_boundary(ggml_tensor * tensor, CaptureState & state, const std::string & owner, const std::string & role) {
    const std::string file = "tensor-"+std::to_string(state.event++)+".bin";
    std::vector<char> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    std::ofstream(state.directory+"/"+file, std::ios::binary).write(bytes.data(), bytes.size());
    std::fprintf(state.manifest,
        "{\"file\":\"%s\",\"owner\":\"%s\",\"role\":\"%s\",\"name\":\"%s\",\"op\":\"%s\",\"type\":%d,\"ne\":[%lld,%lld,%lld,%lld],\"nb\":[%zu,%zu,%zu,%zu],\"bytes\":%zu}\n",
        file.c_str(), owner.c_str(), role.c_str(), tensor->name, ggml_op_name(tensor->op), tensor->type,
        (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3],
        tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3], bytes.size());
    std::fflush(state.manifest);
}
static bool capture_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & state = *static_cast<CaptureState *>(data);
    const bool requested = controlled_precision(tensor, ask, state.precision);
    const std::string name = tensor->name;
    const bool boundary = state.active && capture_boundary(name);
    if (ask) return requested || boundary;
    if (boundary) {
        save_boundary(tensor, state, name, "output");
        const int layer = std::stoi(name.substr(name.rfind('-')+1));
        if (layer < 4 && (tensor->op == GGML_OP_DSV4_HC_POST || tensor->op == GGML_OP_MUL_MAT)) {
            const int count = tensor->op == GGML_OP_DSV4_HC_POST ? 4 : 2;
            for (int i = 0; i < count; ++i) if (tensor->src[i] && ggml_nbytes(tensor->src[i]) <= 4*1024*1024) {
                save_boundary(tensor->src[i], state, name, "src"+std::to_string(i));
            }
        }
    }
    return true;
}
'''
(R/'capture-fragment.cpp').write_text(fragment)
s=s.replace('using namespace ds14;','using namespace ds14;\n'+fragment)
s=s.replace('    cp.cb_eval = controlled_precision;\n    cp.cb_eval_user_data = &options;','    CaptureState state{&options};\n    cp.cb_eval = capture_precision;\n    cp.cb_eval_user_data = &state;')
s=s.replace('        batch.n_tokens = std::min(start < switch_at ? 1 : width, 94-start);','''        batch.n_tokens = std::min(start < switch_at ? 1 : width, 94-start);
        state.active = start == 0 || (start <= 86 && start+batch.n_tokens > 86);
        if (state.active) {
            const int target = start == 0 ? 0 : 86;
            state.directory = out+"/"+label+"-input"+std::to_string(target);
            std::filesystem::create_directories(state.directory);
            state.manifest = std::fopen((state.directory+"/tensors.jsonl").c_str(), "w");
            check(state.manifest, "capture manifest failed");
            std::ofstream meta(state.directory+"/alignment.json");
            meta << "{\\\"input_index\\\":" << target << ",\\\"output_index\\\":" << target+1
                 << ",\\\"batch_start\\\":" << start << ",\\\"width\\\":" << batch.n_tokens
                 << ",\\\"column\\\":" << target-start << "}\\n";
            state.event = 0;
        }''')
s=s.replace('        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);','''        for (int i = 0; i < batch.n_tokens; ++i) record(start+i+1, i);
        if (state.active) { std::fclose(state.manifest); state.manifest = nullptr; }''')
(R/'capture-chain.cpp').write_text(s)
runner=(R.parent/'stage14/run-extended-replay.py').read_text().replace("'extended-replay'","'capture-chain'").replace("'extended-replay-output'","'chain-capture-output'").replace("'extended-replay.cpp'","'capture-chain.cpp'").replace("'extended-replay-manifest.json'","'chain-capture-manifest.json'").replace("'extended-replay.log'","'chain-capture.log'")
runner=runner.replace("all first65 logits must match Stage13; scalar95 argmax must match the native free answer","all95 full logits at widths1/2/4 must match Stage14 extended replay")
runner=runner.replace("after prefill, add scalar router, HC+router, or all ordinary2d floating matmuls to the FP32/scalar FA/upgate control; no production flags","capture input indices0/86 after Stage14 intervention; width4 index86 is column2 of batch84")
(R/'run-chain-capture.py').write_text(runner)
snapshot=R.parent/'stage8/candidate-bin';repo=Path('/home/sergei/Github/llama.cpp')
cmd=['g++','-std=c++17','-O2','-I'+str(repo/'include'),'-I'+str(repo/'src'),'-I'+str(repo/'ggml/include'),str(R/'capture-chain.cpp'),'-L'+str(snapshot),'-Wl,-rpath,'+str(snapshot),'-lllama','-lggml','-lggml-base','-o',str(R/'capture-chain')]
result=subprocess.run(cmd,capture_output=True,text=True)
(R/'build.log').write_text(result.stdout+result.stderr)
(R/'build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':result.returncode,'sources':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['prepare-capture.py','diagnostic-callback.h','capture-fragment.cpp','capture-chain.cpp','run-chain-capture.py']}},indent=2)+'\n')
print(result.returncode,result.stderr)
raise SystemExit(result.returncode)
