from pathlib import Path
import subprocess
R=Path(__file__).resolve().parent;repo=Path('/home/sergei/Github/llama.cpp')
s=subprocess.check_output(['git','show','9df5a4a40:src/llama-memory-hybrid.cpp'],cwd=repo,text=True)
helper='''
static void stage30_layout(const llama_kv_cache * a, const llama_kv_cache * b, const char * phase, llama_seq_id seq, llama_state_seq_flags flags) {
    if (!b || seq < 0) return;
    const auto & ca = a->get_cells(seq);const auto & cb = b->get_cells(seq);
    int mismatch = 0, first = -1;
    for (uint32_t i = 0; i < ca.size(); ++i) {
        if (ca.seq_has(i, seq) != cb.seq_has(i, seq) || (ca.seq_has(i, seq) && ca.pos_get(i) != cb.pos_get(i))) {
            ++mismatch;if(first < 0) first = i;
        }
    }
    LLAMA_LOG_INFO("STAGE30_LAYOUT phase=%s seq=%d flags=%u mismatch=%d first=%d attn_used=%u idx_used=%u amin=%d amax=%d bmin=%d bmax=%d\\n", phase, seq, flags, mismatch, first, ca.get_used(), cb.get_used(), ca.seq_pos_min(seq), ca.seq_pos_max(seq), cb.seq_pos_min(seq), cb.seq_pos_max(seq));
}

'''
s=s.replace('//\n// llama_memory_hybrid\n//',helper+'//\n// llama_memory_hybrid\n//',1)
old='void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {'
s=s.replace(old,old+'\n    stage30_layout(mem_attn.get(), mem_idx.get(), "write", seq_id, flags);')
old='    if (mem_kpool) mem_kpool->invalidate();\n}\n\nllama_kv_cache * llama_memory_hybrid::get_mem_attn()'
s=s.replace(old,'    stage30_layout(mem_attn.get(), mem_idx.get(), "read", seq_id, flags);\n'+old)
(R/'trace-llama-memory-hybrid.cpp').write_text(s)
