from pathlib import Path
import json,shutil
P=Path(__file__).parent;O=Path('/tmp/dsv41-hc-copy-sharing-20261103')
s=(O/'bench.py').read_text();s=s.replace("assert trace['aliases'][1]>0 and trace['aliases'][0]==trace['aliases'][2]==0 and trace['injected_failures']==2", "pool=json.load(open(P/'long-trace.json.pool.json'));assert pool['local'][1]>0 and pool['local'][2]==0 and pool['remote'][1]==0 and pool['remote'][2]>0 and trace['injected_failures']==2")
s=s.replace("e['GGML_SCHED_COPY_RESHAPE']=value", "e['LLAMA_DSV41_CANDIDATE_POOL_LOCAL']=value")
s=s.replace("hc-copy-sharing-screen","candidate-pool-screen").replace("19063","19067").replace("Clean native OFF/ON/OFF,", "Clean native candidate-pool OFF/ON/OFF, HC sharing stays defaultON,")
(P/'bench.py').write_text(s)
# Adapt the bounded read-only transfer observer; keep CUDA arguments and payload untouched.
s=Path('/tmp/dsv41-m2-large-context-20261103/transfer-audit.cpp').read_text()
s=s.replace('return t && (std::strstr(t->name,"l_last-6") || std::strstr(t->name,"l_last-14"));','return t && (std::strstr(t->name,"idx_score-20") || std::strstr(t->name,"candidate_"));')
s=s.replace('wide |= t->ne[2]>6;', 'wide |= t->ne[1]>6;')
s=s.replace('q>=s.data && q-s.data<=s.bytes && bytes<=s.bytes-(q-s.data)', 'q==s.data && bytes==s.bytes')
s=s.replace('{"begin_ns",ns()},{"tensors"', '{"begin_ns",ns()},{"flag",std::getenv("LLAMA_DSV41_CANDIDATE_POOL_LOCAL")?std::getenv("LLAMA_DSV41_CANDIDATE_POOL_LOCAL"):"unset"},{"tensors"')
(P/'transfer-audit.cpp').write_text(s)
# Loader observer is unchanged and rebuilt later only if needed; it allocates no GPU payload.
shutil.copyfile('/tmp/dsv41-hc-on-long-indexer-20261103/placement.cpp',P/'placement.cpp')
