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
original=(REPO/'tools/server/server-context.cpp').read_text();s=original
needle='constexpr int HTTP_POLLING_SECONDS = 1;';assert s.count(needle)==1
s=s.replace(needle,'#include "llama-model.h"\n#include "general-callback.h"\n#include "raw-layout.h"\n\n'+needle)
needle='struct server_context_impl {\n';assert s.count(needle)==1
s=s.replace(needle,needle+'    ds14::PrecisionOptions ds14_options{"decode-scalar-fa-upgate-hc-router-down-compressor", false};\n    ds14::DownState ds17_down{&ds14_options};\n    bool ds14_enabled = false;\n    ds22::RawPlan ds22_plan;\n    int ds22_source_raw = -1;\n    int ds21_r1_dense = 0, ds21_r1_sparse = 0, ds21_r2_dense = 0, ds21_r2_sparse = 0;\n    std::map<ggml_tensor *, ggml_prec> ds14_saved_precision;\n')
needle='        llama_init = common_init_from_params(params_base);';assert s.count(needle)==1
s=s.replace(needle,'''        const char * ds14_env = std::getenv("LLAMA_DSV41_DIAGNOSTIC");
        ds14_enabled = ds14_env && std::string(ds14_env) == "compressor-down";
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
                if (!ask && self.ds14_options.decoding && tensor->op == GGML_OP_FLASH_ATTN_EXT) {
                    const int layer = std::stoi(std::string(tensor->src[0]->name).substr(2));
                    if (layer == 0) {
                        self.ds22_source_raw = tensor->src[1]->ne[1];
                        ds22::check_raw_plan(llama_get_memory(self.ctx_tgt), self.ds22_plan, self.ds22_source_raw);
                    }
                    const int ratio = self.ds14_options.compress_ratios.at(layer);
                    for (int col = 0; col < tensor->src[0]->ne[1]; ++col) {
                        const int p = self.ds14_options.absolute_position + col;
                        const int k = self.ds22_plan.extents[col] + (ratio ? std::max(256, ((p+1)/ratio+255)/256*256) : 0);
                        const bool sparse = tensor->op_params[4] > 0 && k >= std::max(4096, 2*tensor->op_params[4]);
                        if (ratio == 1) ++(sparse ? self.ds21_r1_sparse : self.ds21_r1_dense);
                        if (ratio == 2) ++(sparse ? self.ds21_r2_sparse : self.ds21_r2_dense);
                    }
                }
                return ds14::down_precision(tensor, ask, &self.ds17_down);
            };
            params_base.cb_eval_user_data = this;
        }
'''+needle)
needle='        vocab = llama_model_get_vocab(model_tgt);';assert s.count(needle)==1
s=s.replace(needle,'''        ds14_options.compress_ratios.assign(model_tgt->hparams.dsv4_compress_ratios.begin(),
            model_tgt->hparams.dsv4_compress_ratios.begin() + model_tgt->hparams.n_layer());
'''+needle)
needle='                common_params params_dft = common_base_params_to_speculative(params_base);';assert s.count(needle)==1
s=s.replace(needle,needle+'\n                params_dft.cb_eval = nullptr;\n                params_dft.cb_eval_user_data = nullptr;')
needle='            ret = llama_decode(ctx_tgt, batch_view);';assert s.count(needle)==1
s=s.replace(needle,'''            const bool ds14_generating = std::any_of(slots.begin(), slots.end(), [](const server_slot & slot) {
                return slot.state == SLOT_STATE_GENERATING;
            });
            if (ds14_enabled && ds14_generating) GGML_ASSERT(batch_view.n_tokens <= 4);
            GGML_ASSERT(ds14_saved_precision.empty() && ds17_down.inputs.empty());
            ds14_options.decoding = ds14_enabled && ds14_generating;
            ds14_options.absolute_position = batch_view.pos ? batch_view.pos[0] : -1;
            ds22_source_raw = -1;
            ds22_plan = {};
            ds14_options.raw_query_extents.clear();
            if (ds14_options.decoding) {
                ds22_plan = ds22::raw_plan(llama_get_memory(ctx_tgt), batch_view);
                ds14_options.raw_query_extents = ds22_plan.extents;
            }
            ds21_r1_dense = ds21_r1_sparse = ds21_r2_dense = ds21_r2_sparse = 0;
            const int ds21_comp_crops_before = ds14_options.compressed_padding_crops;
            const int ds18_crops_before = ds14_options.padding_crops;
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
            ds14_saved_precision.clear();\n            ds17_down.inputs.clear();
            ds14_options.decoding = false;
            std::fprintf(stderr, "DS14_EVENT enabled=%d generation=%d width=%d pos=%d ret=%d precision=%d attention=%d upgate=%d matmul=%d padding_crops=%d compressed_crops=%d r1_dense=%d r1_sparse=%d r2_dense=%d r2_sparse=%d raw_source=%d raw_first=%d raw_last=%d physical_first=%d physical_last=%d\\n",
                ds14_enabled, ds14_generating, batch_view.n_tokens, batch_view.pos ? batch_view.pos[0] : -1, ret,
                ds14_options.precision_requests - ds14_precision_before,
                ds14_options.replaced - ds14_attention_before, ds14_options.replaced_upgate - ds14_upgate_before,
                ds14_matmul_total() - ds14_matmul_before, ds14_options.padding_crops - ds18_crops_before,
                ds14_options.compressed_padding_crops - ds21_comp_crops_before, ds21_r1_dense, ds21_r1_sparse, ds21_r2_dense, ds21_r2_sparse,
                ds22_source_raw, ds22_plan.extents.empty() ? -1 : ds22_plan.extents.front(), ds22_plan.extents.empty() ? -1 : ds22_plan.extents.back(),
                ds22_plan.indices.empty() ? -1 : (int) ds22_plan.indices.front(), ds22_plan.indices.empty() ? -1 : (int) ds22_plan.indices.back());
        });''')
(ROOT/'server-context-general.cpp').write_text(s)
(ROOT/'server-general-experiment.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/tools/server/server-context.cpp',tofile='b/tools/server/server-context.cpp')))
(ROOT/'server-build').mkdir(exist_ok=True)
OUT=ROOT/'experiment-bin';
if not OUT.exists():shutil.copytree(SNAP,OUT,symlinks=True)
flags={}
for line in (BUILD/'tools/server/CMakeFiles/server-context.dir/flags.make').read_text().splitlines():
 if line.startswith('CXX_'):key,value=line.split(' = ',1);flags[key]=shlex.split(value)
compile_server=['/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],'-I'+str(REPO/'src'),'-I'+str(ROOT),'-c',str(ROOT/'server-context-general.cpp'),'-o',str(ROOT/'server-build/server-context.cpp.o')]
archive=ROOT/'libserver-context-general.a';shutil.copy2(BUILD/'tools/server/libserver-context.a',archive)
archive_cmd=['/bin/ar','r',str(archive),str(ROOT/'server-build/server-context.cpp.o')]
link=shlex.split((BUILD/'tools/server/CMakeFiles/llama-server-impl.dir/link.txt').read_text())
for i,token in enumerate(link):
 if token=='-o':link[i+1]=str(OUT/'libllama-server-impl.so')
 if token=='libserver-context.a':link[i]=str(archive)
 if token.startswith('-Wl,-rpath,'):link[i]='-Wl,-rpath,'+str(OUT)
 if token.startswith('../../bin/'):link[i]=str(OUT/Path(token).name)
inputs={str(p):sha(p) for p in (BUILD/'tools/server/CMakeFiles/llama-server-impl.dir').glob('*.o')}
inputs.update({str(p):sha(p) for p in [BUILD/'tools/server/libserver-context.a',BUILD/'tools/ui/libllama-ui.a',BUILD/'vendor/cpp-httplib/libcpp-httplib.a',BUILD/'common/libllama-common-base.a']})
meta={'head':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),'commands':[compile_server,archive_cmd,link],'link_cwd':str(BUILD/'tools/server'),'original_server_impl_matches_snapshot':True,'reused_build_inputs':inputs,'sources':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'general-callback.h',ROOT/'raw-layout.h',ROOT/'server-context-general.cpp',ROOT/'server-general-experiment.patch']},'production_change':False,'original_server_source_sha256':sha(REPO/'tools/server/server-context.cpp'),'parent_header_sha256':sha(ROOT.parent/'stage21/general-callback.h'),'down_source_sha256':sha(ROOT.parent/'stage15/down-candidate.cpp')}
(ROOT/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
with (ROOT/'build.txt').open('w') as log:
 for command in meta['commands']:
  result=subprocess.run(command,cwd=BUILD/'tools/server',stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise SystemExit(result.returncode)
meta['exit_code']=0;meta['experiment_hashes']={n:sha(OUT/n) for n in hashes}
assert [n for n in hashes if meta['experiment_hashes'][n]!=hashes[n]]==['libllama-server-impl.so']
(ROOT/'build-manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print('BUILD COMPLETE: only libllama-server-impl.so differs from Stage8',flush=True)
