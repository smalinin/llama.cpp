from pathlib import Path
import subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/sergei/Github/llama.cpp')
s=subprocess.check_output(['git','show','9df5a4a40:src/llama-context.cpp'],cwd=repo,text=True)
helper=r'''
static int stage30_capture_id = -1;
static int stage30_fa_seen = 0;
static bool stage30_capture_active = false;
static void stage30_dump(ggml_tensor * t, const char * name) {
    if (!t) return;
    const char * out = std::getenv("STAGE30_CAPTURE");
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    const std::string file = std::string(out) + "/" + std::to_string(stage30_capture_id) + "-" + name + ".bin";
    FILE * f = fopen(file.c_str(), "wb");GGML_ASSERT(f);fwrite(bytes.data(), 1, bytes.size(), f);fclose(f);
    LLAMA_LOG_INFO("STAGE30_TENSOR id=%d name=%s op=%s type=%d ne=%lld,%lld,%lld,%lld nb=%zu,%zu,%zu,%zu bytes=%zu\n", stage30_capture_id, name, ggml_op_name(t->op), t->type, (long long)t->ne[0],(long long)t->ne[1],(long long)t->ne[2],(long long)t->ne[3],t->nb[0],t->nb[1],t->nb[2],t->nb[3],bytes.size());
}
static bool stage30_capture(ggml_tensor * t, bool ask, void *) {
    if (!std::getenv("STAGE30_CAPTURE") || !stage30_capture_active || stage30_fa_seen || t->op != GGML_OP_FLASH_ATTN_EXT) return false;
    if (ask) return true;
    stage30_dump(t,"fa");
    for (int i=0;i<6;++i) stage30_dump(t->src[i],("src"+std::to_string(i)).c_str());
    const std::string file = std::string(std::getenv("STAGE30_CAPTURE")) + "/" + std::to_string(stage30_capture_id) + "-params.bin";
    FILE * f=fopen(file.c_str(),"wb");GGML_ASSERT(f);fwrite(t->op_params,1,sizeof(t->op_params),f);fclose(f);
    ++stage30_fa_seen;
    return true;
}
'''
s=s.replace('//\n// llama_context\n//',helper+'\n//\n// llama_context\n//',1)
s=s.replace('    cparams.cb_eval           = params.cb_eval;', '    cparams.cb_eval           = std::getenv("STAGE30_CAPTURE") ? stage30_capture : params.cb_eval;')
key='llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {'
s=s.replace(key,key+'''\n    stage30_capture_active = ubatch.n_tokens == 4 && ubatch.pos[0] == 431 && ubatch.seq_id[0][0] == 2;
    stage30_fa_seen = 0;
    if (stage30_capture_active) ++stage30_capture_id;
''')
(r/'fa-capture-llama-context.cpp').write_text(s)
