from pathlib import Path
import subprocess
r=Path(__file__).resolve().parent;repo=Path('/home/sergei/Github/llama.cpp');s=subprocess.check_output(['git','show','9df5a4a40:tools/server/server-context.cpp'],cwd=repo,text=True)
old='''                ret->prompt_save(*prompt_cache);

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {'''
new='''                ret->prompt_save(*prompt_cache);

                // Free idle unified KV cells before restoring a cached prompt.
                if (params_base.cache_idle_slots && params_base.kv_unified && !ret->is_processing()) {
                    for (auto & idle : slots) {
                        if (&idle == ret || idle.is_processing() || idle.prompt.tokens.empty()) {
                            continue;
                        }
                        if (idle.prompt_save(*prompt_cache)) {
                            SLT_DBG(idle, "%s", "__TEST_TAG_CACHE_IDLE_SLOT__\\n");
                            prompt_cache->update();
                        }
                        idle.prompt_clear();
                    }
                }

                if (!ret->prompt_load(*prompt_cache, task.tokens)) {'''
assert s.count(old)==1;s=s.replace(old,new);(r/'preclear-server-context.cpp').write_text(s)
