from pathlib import Path
import subprocess,json,hashlib
R=Path('/home/sergei/Github/llama.cpp');P=Path(__file__).resolve().parent
assert P.parent.name=='1017-verify-float6-acceptance-20261106'
p=P/'lifecycle.cpp';s=p.read_text();d=json.loads((P/'build.json').read_text())
if 'first8K_controls' not in s:
    needle=';out["progress"]={{"corpus",id},{"mode",mode},{"prefix",pos+8192}};'
    assert s.count(needle)==1
    s=s.replace(needle,';if(pos==0){out["first8K_controls"][id]=first;save();}out["progress"]={{"corpus",id},{"mode",mode},{"prefix",pos+8192}};')
    p.write_text(s)
    with(P/'final-build.log').open('wb')as f:subprocess.run(d['commands'][0],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
    d['source_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();d['harness_sha256']=hashlib.sha256((P/'lifecycle-harness').read_bytes()).hexdigest();(P/'build.json').write_text(json.dumps(d,indent=2)+'\n')
else:
    assert hashlib.sha256(p.read_bytes()).hexdigest()==d['source_sha256']
    assert hashlib.sha256((P/'lifecycle-harness').read_bytes()).hexdigest()==d['harness_sha256']
print('CANONICAL_FIXTURE_READY')
