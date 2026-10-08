from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parent
v1=(R/'candidate-callback.h').read_text()
(R/'v1-callback.h').write_text(v1.replace('namespace ds14 {','namespace ds18v1 {'))
s=v1.replace('    int padding_crops = 0;','    int padding_crops = 0;\n    int compressed_padding_crops = 0;')
s=s.replace('            auto crop_kv =', '''            const int compressed = k->ne[1]-raw;
            check(compressed == 0 || compressed == 256 || compressed == 512,"unsupported compressed padding");
            const int keep_comp = std::min(compressed,256);
            if (compressed > keep_comp) {
                std::vector<ggml_fp16_t> comp_tail(compressed-keep_comp);
                ggml_backend_tensor_get(source_mask,comp_tail.data(),col*source_mask->nb[1]+(raw+keep_comp)*2,comp_tail.size()*2);
                for (auto x : comp_tail) check(std::isinf(ggml_fp16_to_fp32(x)) && ggml_fp16_to_fp32(x) < 0,"compressed crop would remove a visible row");
                ++options.compressed_padding_crops;
            }
            auto crop_kv =''')
s=s.replace('leaf->ne[0],leaf->ne[1]-raw,1,1','leaf->ne[0],keep_comp,1,1')
s=s.replace('mask->ne[0]-raw,1,1,1','keep_comp,1,1,1')
s=s.replace('first 256-row raw-cache boundary','first 256-row raw/compressed-cache boundary')
(R/'v2-callback.h').write_text(s)
f=(R/'capture-fragment.cpp').read_text()
a=f.index('    const bool fa =');b=f.index('    if (ask)',a)
f=f[:a]+'''    const bool fa = state.active && tensor->op == GGML_OP_FLASH_ATTN_EXT && std::string(tensor->src[0]->name).find("q-20 ") == 0;
'''+f[b:]
f=f.replace('"FA-" + std::string(tensor->src[0]->name).substr(2,1)','"FA-20"')
(R/'layer20-fragment.cpp').write_text(f)
cap=(R/'candidate-capture.cpp').read_text();cap=cap[cap.index('static void precision_replay'):cap.index('int main(')]
can=(R/'candidate-replay.cpp').read_text();can=can[can.index('static void precision_replay'):can.index('int main(')]
can=can.replace('options.padding_crops);','options.padding_crops);\n    std::printf("COMPRESSED_PADDING_CROPS %s %d\\n",label.c_str(),options.compressed_padding_crops);')
base=(R/'baseline-replay.cpp').read_text();base=base[base.index('static void precision_replay'):base.index('int main(')]
s='#include "v1-callback.h"\n#include "v2-callback.h"\nnamespace v1 {\nusing namespace ds18v1;\n#include "layer20-fragment.cpp"\n'+cap+'}\nnamespace v2 {\nusing namespace ds14;\n'+can+'}\nnamespace baseline {\nusing namespace ds14;\n'+base+'}\nusing namespace ds14;\n'
main=(R/'candidate-replay.cpp').read_text();main=main[main.index('int main('):]
a=main.index('        for (const std::string mode');b=main.index('        llama_model_free',a)
new='''        const std::string mode="decode-scalar-fa-upgate-hc-router-down-compressor";
        const std::string root=argv[4];
        for (const std::string dir : {"v1-layer20", "v2-explain", "v2-baseline"}) std::filesystem::create_directories(root+"/"+dir);
        for (int width : {1,3}) {
            ds18v1::PrecisionOptions options{mode,false};
            v1::precision_replay(model,prompt,forced,width,false,0,root+"/v1-layer20",options);
        }
        for (int width : {1,2,3,4}) {
            PrecisionOptions options{mode,false};
            v2::precision_replay(model,prompt,forced,width,false,0,root+"/v2-explain",options);
        }
        const std::string baseline_text=read_text("BASELINE_PROMPT");
        const int bn=-llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),nullptr,0,true,true);
        std::vector<llama_token> bp(bn),bf(95);
        check(llama_tokenize(vocab,baseline_text.data(),baseline_text.size(),bp.data(),bn,true,true)==bn,"baseline tokenize failed");
        std::ifstream bi("BASELINE_PREFIX",std::ios::binary);
        check(bool(bi.read(reinterpret_cast<char *>(bf.data()),bf.size()*4)),"baseline prefix failed");
        for (int width : {1,2,3,4}) {
            PrecisionOptions options{mode,false};
            baseline::precision_replay(model,bp,bf,width,false,0,root+"/v2-baseline",options);
        }
'''.replace('BASELINE_PROMPT',str(R.parent/'stage7/baseline-prompt.txt')).replace('BASELINE_PREFIX',str(R.parent/'stage9/forced-current-native.i32'))
s+=main[:a]+new+main[b:]
(R/'v2-replay.cpp').write_text(s)
cmd=json.loads((R.parent/'stage17/explain-build-manifest.json').read_text())['command']
old=R.parent/'stage17';cmd=[x.replace(str(old/'explain-replay.cpp'),str(R/'v2-replay.cpp')).replace(str(old/'explain-replay'),str(R/'v2-replay')) for x in cmd]
with (R/'v2-build.txt').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
assert p.returncode==0,(R/'v2-build.txt').read_text()
(R/'v2-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':0,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'v2-replay.cpp',R/'v1-callback.h',R/'v2-callback.h',R/'layer20-fragment.cpp']}},indent=2)+'\n')
s=(R/'run-capture.py').read_text().replace("R/'capture-replay'", "R/'v2-replay'").replace("R/'capture-output'","R/'v2-output'").replace("R/'capture-run-manifest.json'","R/'v2-run-manifest.json'").replace("R/'capture.log'","R/'v2.log'")
(R/'run-v2.py').write_text(s)
