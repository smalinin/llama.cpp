from pathlib import Path
import subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/sergei/Github/llama.cpp')
s=subprocess.check_output(['git','show','9df5a4a40:src/llama-context.cpp'],cwd=repo,text=True)
old='''    return ctx->state_seq_set_data(seq_id, src, size, flags);
}'''
new='''    const size_t loaded = ctx->state_seq_set_data(seq_id, src, size, flags);
    if (loaded && !(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)) {
        const size_t n = ctx->state_seq_get_size(seq_id, flags);
        std::vector<uint8_t> check(n);
        ctx->synchronize();
        const size_t saved = ctx->state_seq_get_data(seq_id, check.data(), n, flags);
        size_t first = 0;
        while (first < std::min(size, n) && src[first] == check[first]) ++first;
        LLAMA_LOG_INFO("STAGE30_ROUNDTRIP seq=%d flags=%u loaded=%zu saved=%zu original=%zu first_diff=%zu equal=%d\\n", seq_id, flags, loaded, saved, size, first, n == size && first == size);
    }
    return loaded;
}'''
assert s.count(old)==1;s=s.replace(old,new)
p=r/'roundtrip-llama-context.cpp'
if p.exists():assert p.read_text()==s
else:p.write_text(s)
