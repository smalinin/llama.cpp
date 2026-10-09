#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess

R=Path(__file__).resolve().parent
OLD=R.parent/'stage24'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

cmd=[p.replace('/stage24/','/stage26/') for p in json.loads((OLD/'replay-build-manifest.json').read_text())['command']]
result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(R/'replay-build.log').write_text(result.stdout)
manifest={'command':cmd,'exit_code':result.returncode,'sources':{n:sha(R/n) for n in ['cache-replay.cpp','general-callback.h','raw-layout.h']},
          'parent_header_sha256':sha(OLD/'general-callback.h'),'parent_raw_layout_sha256':sha(OLD/'raw-layout.h')}
(R/'replay-build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(result.stdout);raise SystemExit(result.returncode)
