import hashlib,json,struct,subprocess,runpy
from pathlib import Path

R=Path(__file__).resolve().parent
helpers=runpy.run_path(str(R/'analyze-attention.py'))
read_tensor=helpers['read_tensor'];capture=helpers['capture']
REPO=Path('/home/sergei/Github/llama.cpp')
SNAP=R.parent/'stage8/candidate-bin'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inp=R/'compressor-inputs'
for name,owner in [('kv','comp_state_kv-2'),('gate','comp_state_score-2')]:
    d=inp/name;d.mkdir(parents=True,exist_ok=True)
    stem=f'blk.2.attn_compressor_{name}.weight'
    meta=json.loads((R.parent/'stage9'/f'{stem}-extraction.json').read_text())
    weight=R.parent/'stage9'/f'{stem}.bf16'
    assert sha(weight)==meta['sha256']
    with open(meta['source'],'rb') as f:
        f.seek(meta['offset']);original=f.read(meta['bytes'])
    assert original==weight.read_bytes()
    (d/'weight.bin').write_bytes(original)
    (d/'config.txt').write_text('30 5120 512\n')
    (d/'selected-columns.txt').write_text('0 0 2\n')
    records=[]
    for w in [1,2,4]:
        dc,ts,c=capture(w,2)
        for suffix,role in [('input','src1'),('captured','output')]:
            t=ts[(owner,role)]
            (d/f'{suffix}-w{w}.f32').write_bytes(read_tensor(dc,t).tobytes())
            records.append({'file':f'{suffix}-w{w}.f32','source':str(dc/t['file']),'source_sha256':sha(dc/t['file']),'tensor':t,'selected_column':c})
    (d/'manifest.json').write_text(json.dumps({'weight':meta,'input_index':2,'inputs':records},indent=2)+'\n')

source=(R.parent/'stage13/isolated-replay.cpp').read_text()
source=source.replace('int n = 5120, m = 5120*4;','int n = 5120, m = 5120*4;\n    int selected[3];\n    std::ifstream selection(root+"/selected-columns.txt");\n    check(bool(selection >> selected[0] >> selected[1] >> selected[2]),"missing columns");')
source=source.replace('for (int source : {1,2,4}) {','for (int source : {1,2,4}) {\n        const int col = selected[source==1 ? 0 : source==2 ? 1 : 2];')
source=source.replace('inputs[0][j];','inputs[0][size_t(col)*n+j];')
source=source.replace('int width, bool actual) {','int width, bool actual, int col) {')
source=source.replace('t, a.data(), c*per_token*4','t, a.data()+size_t(col)*per_token, c*per_token*4')
source=source.replace('width,actual);','width,actual,col);')
source=source.replace('double(result[r])-scalar[r]','double(result[size_t(actual ? col : 0)*m+r])-scalar[r]')
(R/'compressor-replay.cpp').write_text(source)

fa=R/'attention-inputs';fa.mkdir(exist_ok=True)
captures={w:capture(w,2) for w in [1,2,4]}
for w,(dc,ts,c) in captures.items():
    d=fa/f'w{w}';d.mkdir(exist_ok=True)
    axes={'q':1,'mask':1,'out':2,'sinks':None}
    keys={'q':('FA-2','src0'),'mask':('FA-2','src3'),'out':('FA-2','output'),'sinks':('FA-2','src4')}
    for name,key in keys.items():
        a=read_tensor(dc,ts[key],axes[name],c if axes[name] is not None else 0)
        data=b''.join(struct.pack('<e',x) for x in a) if name=='mask' else a.tobytes()
        (d/(name+('.f16' if name=='mask' else '.f32'))).write_bytes(data)
    for name,role in [('k','src1'),('v','src2')]:
        assert ts[('FA-2',role)]['ne']==[512,1792,1,1]
        (d/f'{name}.f16').write_bytes((dc/ts[('FA-2',role)]['file']).read_bytes())
    (d/'manifest.json').write_text(json.dumps({'directory':str(dc),'column':c,'input_index':2,'fa_params':json.loads((dc/'fa-params.json').read_text()),'exact_scale':'C++ 1.0f/std::sqrt(float(512)); JSON scale rounded to 6 decimals is not used','files':{p.name:sha(p) for p in d.iterdir() if p.name!='manifest.json'}},indent=2)+'\n')

ds=fa/'w2-swap';ds.mkdir(exist_ok=True)
for p in (fa/'w1').iterdir():
    if p.name!='manifest.json':(ds/p.name).write_bytes(p.read_bytes())
for name in ['k','v']:
    a=bytearray((fa/'w1'/f'{name}.f16').read_bytes());b=(fa/'w2'/f'{name}.f16').read_bytes()
    a[1596*1024:1597*1024]=b[1596*1024:1597*1024]
    (ds/f'{name}.f16').write_bytes(a)
(ds/'manifest.json').write_text(json.dumps({'base':'w1','replacement':'row1596 from w2 for both K and V','all_query_mask_sinks_unchanged':True,'files':{p.name:sha(p) for p in ds.iterdir() if p.name!='manifest.json'}},indent=2)+'\n')

source=(R.parent/'stage10/attention-replay.cpp').read_text().replace('for (int width : {1,2,3,4,5,8,16})','for (int width : {1})')
source=source.replace('ggml_backend_free(backend);','std::ifstream maps("/proc/self/maps");\n    std::ofstream loaded(output+"/loaded-libraries.txt");\n    for (std::string line; std::getline(maps,line);) if (line.find("libggml") != std::string::npos) loaded << line << char(10);\n    ggml_backend_free(backend);')
(R/'attention-replay.cpp').write_text(source)
build=[]
for name in ['compressor-replay','attention-replay']:
    cmd=['g++','-std=c++17','-O2',f'-I{REPO}/ggml/include',str(R/f'{name}.cpp'),f'-L{SNAP}',f'-Wl,-rpath,{SNAP}','-lggml-cuda','-lggml','-lggml-base','-o',str(R/name)]
    with (R/f'{name}-build.txt').open('w') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    build.append({'command':cmd,'exit_code':proc.returncode,'source_sha256':sha(R/f'{name}.cpp')})
    assert proc.returncode==0
(R/'isolated-build-manifest.json').write_text(json.dumps({'sources':{p:sha(R/p) for p in ['prepare-isolated.py','analyze-chain.py','analyze-attention.py']},'builds':build},indent=2)+'\n')
