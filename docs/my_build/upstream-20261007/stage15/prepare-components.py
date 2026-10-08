from pathlib import Path
import subprocess,json
R=Path(__file__).resolve().parent
s=(R/'down-replay.cpp').read_text()
s=s.replace('check(argc == 3, "usage: down-replay INPUT OUTPUT");','check(argc == 4, "usage: down-components INPUT OUTPUT down|weighted");\n        const std::string stage=argv[3];\n        check(stage=="down" || stage=="weighted", "unknown stage");')
start=s.index('                ggml_build_forward_expand(graph,weighted);')
end=s.index('                auto buffer=',start)
s=s[:start]+'''                auto result=stage=="down" ? down:weighted;
                ggml_build_forward_expand(graph,result);
'''+s[end:]
s=s.replace('std::vector<float> values(5120*width),repeat(values.size());','const int rows=5120*6;\n                std::vector<float> values(rows*width),repeat(values.size());')
s=s.replace('for (int i=0;i<5120;++i) maximum=std::max(maximum,std::abs(double(values[(actual?selected:0)*5120+i])-scalar[i]));','for (int i=0;i<rows;++i) maximum=std::max(maximum,std::abs(double(values[(actual?selected:0)*rows+i])-scalar[i]));')
s=s.replace('for (int col=0;col<width;++col) for (int i=0;i<5120;++i)', 'for (int col=0;col<width;++col) for (int i=0;i<rows;++i)')
s=s.replace('capture_max=std::max(capture_max,std::abs(double(values[col*5120+i])-captured[src*5120+i]));','(void)src;')
s=s.replace('values[col*5120+i])-values[i]', 'values[col*rows+i])-values[i]')
(R/'down-components.cpp').write_text(s)
S=R.parent/'stage8/candidate-bin'
cmd=['g++','-std=c++17','-O2','-I/home/sergei/Github/llama.cpp/ggml/include',str(R/'down-components.cpp'),'-L'+str(S),'-Wl,-rpath,'+str(S),'-lggml-cuda','-lggml','-lggml-base','-o',str(R/'down-components')]
x=subprocess.run(cmd,capture_output=True,text=True)
(R/'components-build.log').write_text(x.stdout+x.stderr)
print(x.returncode,x.stderr)
runner=(R/'run-isolated.py').read_text().replace("['down0']","['down','weighted']").replace("'down-replay'","'down-components'").replace("'down-replay.cpp'","'down-components.cpp'").replace('str(directory),str(output)]','str(directory),str(output),dataset]')
runner=runner.replace("'isolated-profile-manifest.json' if args.profile else 'isolated-manifest.json'","'components-profile-manifest.json' if args.profile else 'components-manifest.json'")
(R/'run-components.py').write_text(runner)
(R/'components-build-manifest.json').write_text(json.dumps({'command':cmd,'exit_code':x.returncode},indent=2)+'\n')
raise SystemExit(x.returncode)
