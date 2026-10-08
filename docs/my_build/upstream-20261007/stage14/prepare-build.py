#!/usr/bin/env python3
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
ROOT=Path(__file__).resolve().parent
REPO=Path('/home/sergei/Github/llama.cpp');BUILD=REPO/'build-glm53';SNAP=ROOT.parent/'stage8/candidate-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text());assert all(sha(SNAP/n)==h for n,h in hashes.items())
assert sha(BUILD/'bin/libllama-server-impl.so')==hashes['libllama-server-impl.so']
base=(ROOT.parent/'stage13/target-matmul-controls.cpp').read_text()
includes=base[:base.index('static void check(')]
helpers=base[base.index('static void check('):base.index('static void precision_replay(')]
helpers=helpers.replace('int replaced_upgate = 0;', 'int replaced_upgate = 0;\n    int precision_requests = 0;')
helpers=helpers.replace('    const std::string name = tensor->name;\n    if (name != "ffn_moe_swiglu_limited-2") return;', '    if (options.directory.empty()) return;\n    const std::string name = tensor->name;\n    if (name != "ffn_moe_swiglu_limited-2") return;')
helpers=helpers.replace('ggml_mul_mat_set_prec(tensor,GGML_PREC_F32);','{ ggml_mul_mat_set_prec(tensor,GGML_PREC_F32); ++options.precision_requests; }')
(ROOT/'diagnostic-callback.h').write_text('#pragma once\n'+includes+'namespace ds14 {\n'+helpers+'}\n')
replay='#include "diagnostic-callback.h"\nusing namespace ds14;\n'+base[base.index('static void precision_replay('):]
replay=replay.replace('{"decode-scalar-fa-upgate-router", "decode-scalar-fa-upgate-hc-router", "decode-scalar-fa-upgate-float2d"}','{"decode-scalar-fa-upgate-hc-router"}')
(ROOT/'replay-control.cpp').write_text(replay)
original=(REPO/'tools/server/server-context.cpp').read_text();s=original
needle='constexpr int HTTP_POLLING_SECONDS = 1;';assert s.count(needle)==1
s=s.replace(needle,'#include "diagnostic-callback.h"\n\n'+needle)
needle='struct server_context_impl {\n';assert s.count(needle)==1
s=s.replace(needle,needle+'    ds14::PrecisionOptions ds14_options{"decode-scalar-fa-upgate-hc-router", false};\n    bool ds14_enabled = false;\n    std::map<ggml_tensor *, ggml_prec> ds14_saved_precision;\n')
needle='        llama_init = common_init_from_params(params_base);';assert s.count(needle)==1
s=s.replace(needle,'''        const char * ds14_env = std::getenv("LLAMA_DSV41_DIAGNOSTIC");
        ds14_enabled = ds14_env && std::string(ds14_env) == "hc-router";
        params_base.n_outputs_max = params_base.n_outputs_max_per_seq = 4;
        if (ds14_enabled) {
            GGML_ASSERT(params_base.n_parallel == 1 && params_base.cb_eval == nullptr);
            params_base.cb_eval = [](ggml_tensor * tensor, bool ask, void * data) {
                auto & self = *static_cast<server_context_impl *>(data);
                if (ask && self.ds14_options.decoding && tensor->op == GGML_OP_MUL_MAT) {
                    const auto type = tensor->src[0]->type;
                    if (type == GGML_TYPE_F32 || type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) {
                        self.ds14_saved_precision.emplace(tensor, static_cast<ggml_prec>(tensor->op_params[0]));
                    }
                }
                return ds14::controlled_precision(tensor, ask, &self.ds14_options);
            };
            params_base.cb_eval_user_data = this;
        }
'''+needle)
needle='                common_params params_dft = common_base_params_to_speculative(params_base);';assert s.count(needle)==1
s=s.replace(needle,needle+'\n                params_dft.cb_eval = nullptr;\n                params_dft.cb_eval_user_data = nullptr;')
needle='            ret = llama_decode(ctx_tgt, batch_view);';assert s.count(needle)==1
s=s.replace(needle,'''            const bool ds14_generating = std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return slot.state == SLOT_STATE_GENERATING;
            });
            if (ds14_enabled && ds14_generating) GGML_ASSERT(batch_view.n_tokens <= 4);
            GGML_ASSERT(ds14_saved_precision.empty());
            ds14_options.decoding = ds14_enabled && ds14_generating;
            const int ds14_attention_before = ds14_options.replaced;
            const int ds14_upgate_before = ds14_options.replaced_upgate;
            const int ds14_precision_before = ds14_options.precision_requests;
            const auto ds14_matmul_total = [&]() {
                int total = 0;
                for (const auto & item : ds14_options.matmul_counts) total += item.second;
                return total;
            };
            const int ds14_matmul_before = ds14_matmul_total();
'''+needle)
needle='                llama_synchronize(ctx_tgt);\n            }\n        });';assert s.count(needle)==1
s=s.replace(needle,'''                llama_synchronize(ctx_tgt);
            }
            if (ret == 0) {
                for (const auto & item : ds14_saved_precision) ggml_mul_mat_set_prec(item.first, item.second);
            }
            ds14_saved_precision.clear();
            ds14_options.decoding = false;
            std::fprintf(stderr, "DS14_EVENT enabled=%d generation=%d width=%d pos=%d ret=%d precision=%d attention=%d upgate=%d matmul=%d\\n",
                ds14_enabled, ds14_generating, batch_view.n_tokens, batch_view.pos ? batch_view.pos[0] : -1, ret,
                ds14_options.precision_requests - ds14_precision_before,
                ds14_options.replaced - ds14_attention_before, ds14_options.replaced_upgate - ds14_upgate_before,
                ds14_matmul_total() - ds14_matmul_before);
        });''')
(ROOT/'server-context.cpp').write_text(s)
(ROOT/'server-experiment.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/tools/server/server-context.cpp',tofile='b/tools/server/server-context.cpp')))
OUT=ROOT/'experiment-bin';
if not OUT.exists():shutil.copytree(SNAP,OUT,symlinks=True)
flags={}
for line in (BUILD/'tools/server/CMakeFiles/server-context.dir/flags.make').read_text().splitlines():
 if line.startswith('CXX_'):key,value=line.split(' = ',1);flags[key]=shlex.split(value)
compile_server=['/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(REPO/'src'),'-I'+str(ROOT),'-c',str(ROOT/'server-context.cpp'),'-o',str(ROOT/'server-context.cpp.o')]
archive=ROOT/'libserver-context.a';shutil.copy2(BUILD/'tools/server/libserver-context.a',archive)
archive_cmd=['/bin/ar','r',str(archive),str(ROOT/'server-context.cpp.o')]
link=shlex.split((BUILD/'tools/server/CMakeFiles/llama-server-impl.dir/link.txt').read_text())
for i,token in enumerate(link):
 if token=='-o':link[i+1]=str(OUT/'libllama-server-impl.so')
 if token=='libserver-context.a':link[i]=str(archive)
 if token.startswith('-Wl,-rpath,'):link[i]='-Wl,-rpath,'+str(OUT)
 if token.startswith('../../bin/'):link[i]=str(OUT/Path(token).name)
replay_cmd=['g++','-std=c++17','-O2','-I'+str(REPO/'include'),'-I'+str(REPO/'src'),'-I'+str(REPO/'ggml/include'),str(ROOT/'replay-control.cpp'),'-L'+str(SNAP),'-Wl,-rpath,'+str(SNAP),'-lllama','-lggml','-lggml-base','-o',str(ROOT/'replay-control')]
inputs={str(p):sha(p) for p in (BUILD/'tools/server/CMakeFiles/llama-server-impl.dir').glob('*.o')}
inputs.update({str(p):sha(p) for p in [BUILD/'tools/server/libserver-context.a',BUILD/'tools/ui/libllama-ui.a',BUILD/'vendor/cpp-httplib/libcpp-httplib.a',BUILD/'common/libllama-common-base.a']})
meta={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'commands':[compile_server,archive_cmd,link,replay_cmd],'link_cwd':str(BUILD/'tools/server'),'original_server_impl_matches_snapshot':True,'reused_build_inputs':inputs,'sources':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'diagnostic-callback.h',ROOT/'server-context.cpp',ROOT/'server-experiment.patch',ROOT/'replay-control.cpp']},'production_change':False}
(ROOT/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
with (ROOT/'build.log').open('w') as log:
 for command in meta['commands']:
  result=subprocess.run(command,cwd=BUILD/'tools/server',stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise SystemExit(result.returncode)
meta['exit_code']=0;meta['experiment_hashes']={n:sha(OUT/n) for n in hashes};meta['replay_binary_sha256']=sha(ROOT/'replay-control')
assert [n for n in hashes if meta['experiment_hashes'][n]!=hashes[n]]==['libllama-server-impl.so']
(ROOT/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print('BUILD COMPLETE: only libllama-server-impl.so differs from Stage8',flush=True)
