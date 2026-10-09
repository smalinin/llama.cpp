#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,struct

R=Path(__file__).resolve().parent
NATIVE=R.parent/'stage25/free-runs/snapshot-off'
INDEX=json.loads((R.parent/'stage25/raw-file-sha256.json').read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

(R/'inputs').mkdir(exist_ok=True)
manifest={}
for kind in ['fresh','cache']:
    req=NATIVE/f'ratio2-long-{kind}-request.json';resp=NATIVE/f'ratio2-long-{kind}-response.json'
    for p in [req,resp]:assert sha(p)==INDEX[str(p.relative_to(R.parent/'stage25'))]['sha256']
    request=json.loads(req.read_text());response=json.loads(resp.read_text())
    assert len(request['prompt'])==6617 and len(response['tokens'])==1024 and request['ignore_eos']
    for name,ids in [('prompt',request['prompt']),(kind,response['tokens'])]:
        p=R/'inputs'/f'{name}.i32';data=struct.pack('<'+'i'*len(ids),*ids)
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
        manifest[p.name]={'tokens':len(ids),'sha256':sha(p)}
    manifest[kind+'_reference']={'request':str(req),'response':str(resp),'request_sha256':sha(req),'response_sha256':sha(resp),'ignore_eos':True}
(R/'inputs/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Native prompt and histories verified')
