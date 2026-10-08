from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parent
fragment=(R/'scalar-down-fragment.cpp').read_text()
fragment=fragment.replace('PrecisionOptions & options)', 'PrecisionOptions & options, const DownInput & input)')
start=fragment.index('    auto * h=borrow(hidden);');end=fragment.index('    auto * w=borrow(down->src[0]);',start)
fragment=fragment[:start]+'''    auto * h=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,hidden->ne[0],6,width);
    auto * all_ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,6,width);
    auto * all_weights=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,6,width);
'''+fragment[end:]
fragment=fragment.replace('    check(buffer,"scalar down allocation failed");','''    check(buffer,"scalar down allocation failed");
    check(input.hidden.size()==size_t(hidden->ne[0]*6*width) && input.ids.size()==size_t(6*width) && input.weights.size()==size_t(6*width),"cached down inputs incomplete");
    ggml_backend_tensor_set(h,input.hidden.data(),0,input.hidden.size()*4);
    ggml_backend_tensor_set(all_ids,input.ids.data(),0,input.ids.size()*4);
    ggml_backend_tensor_set(all_weights,input.weights.data(),0,input.weights.size()*4);''')
(R/'scalar-down-cached.cpp').write_text(fragment)
prefix=r'''
struct DownInput {
    std::vector<float> hidden,weights;
    std::vector<int32_t> ids;
};
struct DownState {
    PrecisionOptions * precision;
    std::map<int,DownInput> inputs;
    std::string capture_prefix;
};
'''
suffix=r'''
static bool down_precision(ggml_tensor * tensor, bool ask, void * data) {
    auto & state=*static_cast<DownState *>(data);
    auto & options=*state.precision;
    const bool requested=controlled_precision(tensor,ask,&options);
    if (!options.decoding) return requested;
    const std::string name=tensor->name;
    const bool weight=name.find("ffn_moe_weights_scaled-")==0;
    const bool hidden=name.find("ffn_moe_swiglu_limited-")==0;
    const bool output=name.find("ffn_moe_out-")==0;
    const bool capture=!state.capture_prefix.empty() && (name=="ffn_moe_out-0" || name=="ffn_out-0");
    if (ask) return requested || weight || output || capture;
    if (weight || hidden || output) {
        const int layer=std::stoi(name.substr(name.rfind('-')+1));
        auto & input=state.inputs[layer];
        if (weight && tensor->ne[2]>1) {
            check(ggml_is_contiguous(tensor),"non-contiguous expert weights");
            input.weights.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor,input.weights.data(),0,input.weights.size()*4);
        }
        if (hidden && tensor->ne[2]>1) {
            check(ggml_is_contiguous(tensor),"non-contiguous hidden");
            input.hidden.resize(ggml_nelements(tensor));
            ggml_backend_tensor_get(tensor,input.hidden.data(),0,input.hidden.size()*4);
            auto * ids=tensor->src[0]->src[2];
            check(ids && ids->type==GGML_TYPE_I32 && ids->ne[0]==6,"unsupported cached IDs");
            input.ids.resize(6*tensor->ne[2]);
            for (int col=0;col<tensor->ne[2];++col) ggml_backend_tensor_get(ids,input.ids.data()+6*col,col*ids->nb[1],6*4);
        }
        if (output && tensor->ne[1]>1) scalar_routed_down(tensor,options,input);
    }
    if (capture) {
        std::vector<float> values(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor,values.data(),0,values.size()*4);
        std::ofstream(state.capture_prefix+"-"+name+".f32",std::ios::binary).write(reinterpret_cast<const char *>(values.data()),values.size()*4);
    }
    return true;
}
'''
s=(R.parent/'stage14/extended-replay.cpp').read_text().replace('using namespace ds14;','using namespace ds14;\n'+prefix+fragment+suffix)
s=s.replace('    cp.cb_eval = controlled_precision;\n    cp.cb_eval_user_data = &options;','    DownState state{&options};\n    cp.cb_eval = down_precision;\n    cp.cb_eval_user_data = &state;')
s=s.replace('        batch.n_tokens = std::min(start < switch_at ? 1 : width, 94-start);','''        batch.n_tokens = std::min(start < switch_at ? 1 : width, 94-start);
        const int target=start==0?0:86;
        state.capture_prefix=(start==0 || (start<=86 && start+batch.n_tokens>86)) ? out+"/"+label+"-input"+std::to_string(target):"";''')
s=s.replace('"decode-scalar-fa-upgate-hc-router"','"decode-scalar-fa-upgate-hc-router-down"')
(R/'down-candidate.cpp').write_text(s)
S=R.parent/'stage8/candidate-bin';repo=Path('/home/sergei/Github/llama.cpp')
cmd=['g++','-std=c++17','-O2',*[ '-I'+str(repo/p) for p in ['include','src','ggml/include']],str(R/'down-candidate.cpp'),'-L'+str(S),'-Wl,-rpath,'+str(S),'-lllama','-lggml','-lggml-base','-o',str(R/'down-candidate')]
x=subprocess.run(cmd,capture_output=True,text=True);(R/'candidate-build.log').write_text(x.stdout+x.stderr)
print(x.returncode,x.stderr)
runner=(R/'run-chain-capture.py').read_text().replace("'capture-chain'","'down-candidate'").replace("'chain-capture-output'","'down-candidate-output'").replace("'capture-chain.cpp'","'down-candidate.cpp'").replace("'chain-capture-manifest.json'","'down-candidate-manifest.json'").replace("'chain-capture.log'","'down-candidate.log'")
runner=runner.replace("all95 full logits at widths1/2/4 must match Stage14 extended replay","scalar95 logits must match Stage14; measure whether scalar routed down/weight/sum resolves wide differences")
runner=runner.replace("capture input indices0/86 after Stage14 intervention; width4 index86 is column2 of batch84","after Stage14 intervention, scalarize routed down/weight/sum using live cached hidden, IDs and weights; no production change")
runner=runner.replace("['decode-scalar-fa-upgate-hc-router']","['decode-scalar-fa-upgate-hc-router-down']")
(R/'run-down-candidate.py').write_text(runner)
(R/'candidate-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':x.returncode,'sources':{f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in ['prepare-candidate.py','scalar-down-cached.cpp','down-candidate.cpp','run-down-candidate.py','diagnostic-callback.h']}},indent=2)+'\n')
raise SystemExit(x.returncode)
