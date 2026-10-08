from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parent
old=R.parent/'stage18'
s=(old/'v2-callback.h').read_text()
(R/'legacy-callback.h').write_text(s.replace('namespace ds14 {','namespace ds19legacy {'))
s=s.replace('    int raw_capacity = 0;', '    int raw_capacity = 0;\n    int raw_limit = 0;\n    std::vector<int> compress_ratios;')
s=s.replace('    if (std::string(source_q->name).find("q-0 ") == 0) options.raw_capacity = tensor->src[1]->ne[1];', '''    const int layer = std::stoi(std::string(source_q->name).substr(2));
    check(layer >= 0 && layer < (int) options.compress_ratios.size(),"missing compression ratio");
    if (layer == 0) {
        options.raw_capacity = tensor->src[1]->ne[1];
        options.raw_limit = tensor->src[1]->nb[3]/tensor->src[1]->nb[1];
        check(options.raw_limit >= options.raw_capacity && options.raw_limit%256 == 0,"unsupported raw storage");
    }''')
a=s.index('        // This probe');b=s.index('        auto * y = ggml_flash_attn_ext',a)
s=s[:a]+'''        // This probe assumes a fresh, append-only sequence with no compaction.
        check(options.absolute_position >= 0,"missing absolute query position");
        const int pos = options.absolute_position+col;
        const int raw = options.raw_capacity;
        const int compressed = k->ne[1]-raw;
        const int ratio = options.compress_ratios[layer];
        const auto pad256 = [](int n) { return std::max(256,((n+255)/256)*256); };
        const int keep_raw = std::min(options.raw_limit,pad256(pos+1));
        const int keep_comp = ratio ? pad256((pos+1)/ratio) : 0;
        check(k->ne[2] == 1 && v->ne[2] == 1 && raw >= keep_raw && compressed >= keep_comp,"unsupported cache extent");
        check(mask->ne[0] == raw+compressed,"mask extent mismatch");
        if (raw != keep_raw || compressed != keep_comp) {
            auto check_tail = [&](int start, int count) {
                std::vector<ggml_fp16_t> tail(count);
                if (count) ggml_backend_tensor_get(source_mask,tail.data(),col*source_mask->nb[1]+start*2,count*2);
                for (auto x : tail) check(std::isinf(ggml_fp16_to_fp32(x)) && ggml_fp16_to_fp32(x) < 0,"crop would remove a visible row");
            };
            check_tail(keep_raw,raw-keep_raw);
            check_tail(raw+keep_comp,compressed-keep_comp);
            auto crop_kv = [&](ggml_tensor * leaf) {
                auto * first = ggml_view_4d(ctx,leaf,leaf->ne[0],keep_raw,1,1,leaf->nb[1],leaf->nb[2],leaf->nb[3],0);
                if (!keep_comp) return first;
                auto * rest = ggml_view_4d(ctx,leaf,leaf->ne[0],keep_comp,1,1,leaf->nb[1],leaf->nb[2],leaf->nb[3],raw*leaf->nb[1]);
                return ggml_concat(ctx,first,rest,1);
            };
            k = crop_kv(k);
            v = tensor->src[1] == tensor->src[2] ? k : crop_kv(v);
            auto * first = ggml_view_4d(ctx,mask,keep_raw,1,1,1,mask->nb[1],mask->nb[2],mask->nb[3],0);
            if (!keep_comp) mask = first;
            else {
                auto * rest = ggml_view_4d(ctx,mask,keep_comp,1,1,1,mask->nb[1],mask->nb[2],mask->nb[3],raw*mask->nb[0]);
                mask = ggml_concat(ctx,first,rest,0);
            }
            if (raw != keep_raw) ++options.padding_crops;
            if (compressed != keep_comp) ++options.compressed_padding_crops;
        }
'''+s[b:]
(R/'general-callback.h').write_text(s)
# Keep the input prefix fixed while crossing more cache boundaries.
source=R.parent/'stage17/free-runs/snapshot-off'
tokens=[];pieces=[]
for name in ['explain','svg','python','explain']:
 p=source/f'{name}-greedy-2-response.json';v=json.loads(p.read_text());assert len(v['tokens'])==256
 tokens.extend(v['tokens']);pieces.append({'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'tokens':256})
import struct
(R/'history.i32').write_bytes(struct.pack('<1024i',*tokens))
(R/'history-manifest.json').write_text(json.dumps({'kind':'Teacher-forced concatenation of native response token IDs; not free generation or a quality benchmark','pieces':pieces,'tokens':1024,'sha256':hashlib.sha256((R/'history.i32').read_bytes()).hexdigest()},indent=2)+'\n')
