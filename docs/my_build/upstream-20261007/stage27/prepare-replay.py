#!/usr/bin/env python3
from pathlib import Path
import difflib,hashlib,json,shutil,subprocess
R=Path(__file__).resolve().parent;OLD=R.parent/'stage26';REPO=Path('/home/sergei/Github/llama.cpp')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for name in ['general-callback.h','raw-layout.h']:shutil.copy2(OLD/name,R/name)
source=(OLD/'verify-replay.cpp').read_text().replace('check_schedule_layout(state,e);','')
(R/'rollback-replay.cpp').write_text(source)
fragment=(OLD/'schedule-fragment.h').read_text();a=fragment.index('static void check_schedule_layout(');b=fragment.index('static void warm_schedule(',a)
fragment=fragment[:a]+fragment[b:];fragment=fragment.replace('check_schedule_layout(state,e);','')
(R/'schedule-fragment.h').write_text(fragment)
patch=''
for before,after in [('verify-replay.cpp','rollback-replay.cpp'),('schedule-fragment.h','schedule-fragment.h')]:
 patch+=''.join(difflib.unified_diff((OLD/before).read_text().splitlines(True),(R/after).read_text().splitlines(True),fromfile='stage26/'+before,tofile='stage27/'+after))
(R/'replay.patch').write_text(patch)
rows=(OLD/'verify-cases.txt').read_text().splitlines();cases=rows[:2]
cases += [line for line in (OLD/'cases.txt').read_text().splitlines() if line.split()[0] in ['fresh-native-w1','cache-native-w1']]
cases += [line for line in rows if line.split()[0] in ['fresh-actual','cache-actual']]
assert len(cases)==6;(R/'cases.txt').write_text('\n'.join(cases)+'\n')
cmd=['g++','-std=c++17','-O2',f'-I{REPO}/include',f'-I{REPO}/src',f'-I{REPO}/ggml/include',str(R/'rollback-replay.cpp'),f'-L{R}/candidate-bin',f'-Wl,-rpath,{R}/candidate-bin','-lllama','-lggml','-lggml-base','-o',str(R/'rollback-replay')]
p=subprocess.run(cmd,capture_output=True,text=True);(R/'replay-build.log').write_text(p.stdout+p.stderr)
manifest={'command':cmd,'exit_code':p.returncode,'sources':{n:sha(R/n) for n in ['rollback-replay.cpp','schedule-fragment.h','general-callback.h','raw-layout.h']}}
(R/'replay-build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');assert p.returncode==0,p.stderr
s=(OLD/'run-verification.py').read_text().replace("S=R.parent/'stage8/candidate-bin'","S=R/'candidate-bin'")
s=s.replace("json.loads((R.parent/'stage8/candidate-binary-sha256.json').read_text())","json.loads((R/'build-manifest.json').read_text())['candidate_hashes']")
s=s.replace("R/'verify-build-manifest.json'","R/'replay-build-manifest.json'").replace("R/'verify-replay'","R/'rollback-replay'")
s=s.replace("'production_change':False","'production_change':True,'source_sha256':sha('/home/sergei/Github/llama.cpp/src/llama-kv-cache.cpp')")
s=s.replace("print('START model replay',flush=True)","print('START model replay',flush=True)\nassert m['source_sha256']==json.loads((R/'build-manifest.json').read_text())['source_sha256']")
(R/'run-replay.py').write_text(s)
print('Replay built; old server-layout equality guard replaced by comparison with scalar traces during analysis')
