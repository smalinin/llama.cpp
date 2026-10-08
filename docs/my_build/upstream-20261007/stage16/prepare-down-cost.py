from pathlib import Path
import json,subprocess,hashlib
R=Path(__file__).resolve().parent
s=(R.parent/'stage15/down-replay.cpp').read_text().replace('#include <cmath>','#include <cmath>\n#include <chrono>')
s=s.replace('std::ofstream report(output+"/results.jsonl");report<<std::setprecision(17);','std::ofstream report(output+"/results.jsonl");report<<std::setprecision(17);\n        std::ofstream timings(output+"/timings.jsonl");timings<<std::setprecision(17);')
s=s.replace('for (int width:{1,2,4}) for (bool actual:{false,true}) {','for (int width:{1,2,4}) for (bool actual:{false,true}) for (const std::string mode:{"native","scalar-down"}) {')
s=s.replace('const std::string label="input"','const std::string label=mode+"-input"')
start=s.index('                auto down=ggml_mul_mat_id(');end=s.index('                auto buffer=',start)
s=s[:start]+'''                auto project=[&](ggml_tensor * h,ggml_tensor * id,ggml_tensor * ew,int tokens) {
                    auto down=ggml_mul_mat_id(ctx,weight,h,id);
                    auto weighted=ggml_mul(ctx,down,ew);
                    ggml_build_forward_expand(graph,weighted);
                    std::vector<ggml_tensor *> views;
                    for (int i=0;i<6;++i) {
                        auto view=ggml_view_2d(ctx,weighted,5120,tokens,weighted->nb[2],i*weighted->nb[1]);
                        ggml_build_forward_expand(graph,view);views.push_back(view);
                    }
                    auto result=views[0];
                    for (int i=1;i<6;++i) {result=ggml_add(ctx,result,views[i]);ggml_build_forward_expand(graph,result);}
                    return result;
                };
                ggml_tensor * result=nullptr;
                if (mode=="native") result=project(hidden,ids,weights,width);
                else for (int col=0;col<width;++col) {
                    auto h=ggml_view_3d(ctx,hidden,2304,6,1,hidden->nb[1],hidden->nb[2],col*hidden->nb[2]);
                    auto id=ggml_view_2d(ctx,ids,6,1,ids->nb[1],col*ids->nb[1]);
                    auto ew=ggml_view_3d(ctx,weights,1,6,1,weights->nb[1],weights->nb[2],col*weights->nb[2]);
                    auto y=project(h,id,ew,1);
                    result=result?ggml_concat(ctx,result,y,1):y;
                }
                ggml_build_forward_expand(graph,result);
'''+s[end:]
s=s.replace('check(values==repeat,"repeat changed");','check(std::memcmp(values.data(),repeat.data(),values.size()*4)==0,"repeat changed");')
s=s.replace('                write(output+"/"+label+".f32",values);','''                if (source==1 && target==0 && !actual) {
                    for (int warm=0;warm<10;++warm) check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"warmup failed");
                    ggml_backend_synchronize(backend);
                    for (int group=0;group<5;++group) {
                        const auto start=std::chrono::steady_clock::now();
                        for (int attempt=0;attempt<100;++attempt) check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"benchmark failed");
                        ggml_backend_synchronize(backend);
                        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100;
                        timings<<"{\\\"mode\\\":\\\""<<mode<<"\\\",\\\"width\\\":"<<width<<",\\\"group\\\":"<<group<<",\\\"iterations\\\":100,\\\"mean_ms\\\":"<<ms<<"}\\n";timings.flush();
                    }
                }
                write(output+"/"+label+".f32",values);''')
(R/'down-cost.cpp').write_text(s)
S=R.parent/'stage8/candidate-bin';cmd=['g++','-std=c++17','-O2','-I/home/sergei/Github/llama.cpp/ggml/include',str(R/'down-cost.cpp'),'-L'+str(S),'-Wl,-rpath,'+str(S),'-lggml-cuda','-lggml','-lggml-base','-o',str(R/'down-cost')]
x=subprocess.run(cmd,capture_output=True,text=True);(R/'down-cost-build.txt').write_text(x.stdout+x.stderr)
(R/'down-cost-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':x.returncode,'source_sha256':hashlib.sha256((R/'down-cost.cpp').read_bytes()).hexdigest()},indent=2)+'\n');print(x.returncode,x.stderr)
runner=(R.parent/'stage15/run-isolated.py').read_text().replace("'down-replay.cpp'","'down-cost.cpp'").replace("'down-replay'","'down-cost'").replace("directory=ROOT/'inputs'","directory=ROOT.parent/'stage15/inputs'").replace("['down0']","['down-cost']")
(R/'run-down-cost.py').write_text(runner)
raise SystemExit(x.returncode)
