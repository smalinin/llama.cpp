from pathlib import Path
import json,subprocess,hashlib
R=Path(__file__).resolve().parent
old=R.parent/'stage17'
s=(old/'diagnostic-callback.h').read_text()
s=s.replace('    int precision_requests = 0;', '    int precision_requests = 0;\n    int absolute_position = -1;\n    int raw_capacity = 0;\n    int padding_crops = 0;')
s=s.replace('    for (int col = 0; col < source_q->ne[1]; ++col) {','''    if (std::string(source_q->name).find("q-0 ") == 0) options.raw_capacity = tensor->src[1]->ne[1];
    for (int col = 0; col < source_q->ne[1]; ++col) {''',1)
s=s.replace('        auto * y = ggml_flash_attn_ext(ctx, q, borrow(tensor->src[1]), borrow(tensor->src[2]), mask, scale, max_bias, cap);', '''        auto * k = borrow(tensor->src[1]);
        auto * v = borrow(tensor->src[2]);
        // This probe only covers the first 256-row raw-cache boundary.
        if (options.absolute_position >= 0 && options.absolute_position+col < 256 && options.raw_capacity > 256) {
            const int raw = options.raw_capacity;
            check(raw == 512 && k->ne[2] == 1 && v->ne[2] == 1, "unsupported padding boundary");
            std::vector<ggml_fp16_t> tail(raw-256);
            ggml_backend_tensor_get(source_mask,tail.data(),col*source_mask->nb[1]+256*2,tail.size()*2);
            for (auto x : tail) check(std::isinf(ggml_fp16_to_fp32(x)) && ggml_fp16_to_fp32(x) < 0,"crop would remove a visible row");
            auto crop_kv = [&](ggml_tensor * leaf) {
                auto * first = ggml_view_4d(ctx,leaf,leaf->ne[0],256,1,1,leaf->nb[1],leaf->nb[2],leaf->nb[3],0);
                if (leaf->ne[1] == raw) return first;
                auto * rest = ggml_view_4d(ctx,leaf,leaf->ne[0],leaf->ne[1]-raw,1,1,leaf->nb[1],leaf->nb[2],leaf->nb[3],raw*leaf->nb[1]);
                return ggml_concat(ctx,first,rest,1);
            };
            k = crop_kv(k);
            v = tensor->src[1] == tensor->src[2] ? k : crop_kv(v);
            auto * first = ggml_view_4d(ctx,mask,256,1,1,1,mask->nb[1],mask->nb[2],mask->nb[3],0);
            if (mask->ne[0] == raw) mask = first;
            else {
                auto * rest = ggml_view_4d(ctx,mask,mask->ne[0]-raw,1,1,1,mask->nb[1],mask->nb[2],mask->nb[3],raw*mask->nb[0]);
                mask = ggml_concat(ctx,first,rest,0);
            }
            ++options.padding_crops;
        }
        auto * y = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, cap);''')
s=s.replace('ggml_graph_n_nodes(graph) <= 3','ggml_graph_n_nodes(graph) <= 16',1)
(R/'candidate-callback.h').write_text(s)
s=(old/'explain-replay.cpp').read_text().replace('"diagnostic-callback.h"','"candidate-callback.h"')
s=s.replace('        const int target=start==0?0:86;', '        options.absolute_position = prompt.size()+start;')
s=s.replace(', "decode-scalar-fa-upgate-hc-router-down-float2d"','')
s=s.replace('    std::printf("DONE %s rows=256 vocab=%d prompt=%zu\\n", label.c_str(), vocab, prompt.size());','    std::printf("PADDING_CROPS %s %d\\n",label.c_str(),options.padding_crops);\n    std::printf("DONE %s rows=256 vocab=%d prompt=%zu\\n", label.c_str(), vocab, prompt.size());')
(R/'candidate-replay.cpp').write_text(s)
base=(old/'replay-control.cpp').read_text().replace('"diagnostic-callback.h"','"candidate-callback.h"')
base=base.replace('        const int target=start==0?0:86;','        const int target=start==0?0:86;\n        options.absolute_position = prompt.size()+start;')
(R/'baseline-replay.cpp').write_text(base)
commands=[]
for name in ['candidate','baseline']:
 cmd=json.loads((old/'explain-build-manifest.json').read_text())['command']
 cmd=[x.replace(str(old/'explain-replay.cpp'),str(R/f'{name}-replay.cpp')).replace(str(old/'explain-replay'),str(R/f'{name}-replay')) for x in cmd]
 commands.append(cmd)
 with (R/f'{name}-build.txt').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert result.returncode==0,(R/f'{name}-build.txt').read_text()
(R/'candidate-build-manifest.json').write_text(json.dumps({'commands':commands,'exit_code':0,'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'candidate-callback.h',R/'candidate-replay.cpp',R/'baseline-replay.cpp']}},indent=2)+'\n')
