#!/usr/bin/env python3
from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parent
s=(R/'cache-replay.cpp').read_text()
s=s.replace('struct EvalState {','''static bool focused_boundary(const std::string & name) {
    if (name.size()>=3 && (name.substr(name.size()-3)=="-24" || name.find("-24 ")!=std::string::npos)) return true;
    for (int layer=20;layer<=24;++layer) {
        if (name.find("idx_")==0 && name.size()>=3 && name.substr(name.size()-3)=="-"+std::to_string(layer)) return true;
    }
    return false;
}
struct EvalState {''')
s=s.replace('capture_boundary(tensor->name)','focused_boundary(tensor->name)').replace('std::stoi(std::string(tensor->src[0]->name).substr(2)) < 3','std::stoi(std::string(tensor->src[0]->name).substr(2)) == 24')
s=s.replace('if (boundary) save_boundary(tensor,capture,tensor->name,"output");','''if (boundary) {
        save_boundary(tensor,capture,tensor->name,"output");
        const std::string name=tensor->name;
        if (name.find("idx_score-")==0 || name=="attn_wo_a-24" || name=="attn_out-24") {
            for(int i=0;i<4;++i) if(tensor->src[i]) save_boundary(tensor->src[i],capture,name,"src"+std::to_string(i));
        }
    }''')
s=s.replace('for (int i=0;i<5;++i) if (tensor->src[i]) save_boundary(tensor->src[i],capture,owner,"src"+std::to_string(i));','''for (int i=0;i<5;++i) if (tensor->src[i]) save_boundary(tensor->src[i],capture,owner,"src"+std::to_string(i));
        std::ofstream params(capture.directory+"/"+owner+"-params.bin",std::ios::binary);
        params.write(reinterpret_cast<char *>(tensor->op_params),sizeof(tensor->op_params));''')
s=s.replace('loaded.close();','''loaded.close();
        std::ofstream hp(std::string(argv[3])+"/model-sources.json");hp<<"[";
        for(int il=0;il<model->hparams.n_layer();++il) hp<<(il?",":"")<<"{\\"layer\\":"<<il<<",\\"ratio\\":"<<model->hparams.dsv4_compress_ratios[il]<<",\\"kv_source\\":"<<model->hparams.dsv41_kv_source[il]<<",\\"key_source\\":"<<model->hparams.dsv41_index_key_source[il]<<",\\"topk_source\\":"<<model->hparams.dsv41_topk_source[il]<<"}";
        hp<<"]\\n";hp.close();''')
s=s.replace('capture.active=(start<=749 && 749<start+n) || (start<=750 && 750<start+n);','capture.active=directory.find("fresh-control-")==std::string::npos && ((start<=749 && 749<start+n) || (start<=750 && 750<start+n));')
(R/'focused-replay.cpp').write_text(s)
runner=(R/'run-replay.py').read_text().replace("R/'cache-replay'","R/'focused-replay'")
(R/'run-focused.py').write_text(runner)
lines=(R/'cases.txt').read_text().splitlines()
focused=[line for line in lines if line.split()[0] in ['cache-history-w1','cache-history-w4']]
focused += [line.replace('fresh-history-','fresh-control-') for line in lines if line.startswith('fresh-history-')]
(R/'focused-cases.txt').write_text('\n'.join(focused)+'\n')
m=json.loads((R/'replay-build-manifest.json').read_text());m['command']=[p.replace('/cache-replay','/focused-replay') for p in m['command']]
p=subprocess.run(m['command'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);m['exit_code']=p.returncode
m['sources']={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['focused-replay.cpp','capture-fragment.h','general-callback.h','raw-layout.h']}
(R/'focused-build-manifest.json').write_text(json.dumps(m,indent=2)+'\n');(R/'focused-build.log').write_text(p.stdout);print(p.stdout);print('focused build',p.returncode);raise SystemExit(p.returncode)
