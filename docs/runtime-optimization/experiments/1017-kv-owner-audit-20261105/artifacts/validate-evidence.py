from pathlib import Path
import json,hashlib,re,subprocess,shutil
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-kv-owner-audit-20261105';P=E/'artifacts';D=R/'docs/runtime-optimization/designs/1017-prefill-cache-deferral.md'
for p in [E/'REPORT.md',D]:
 for u in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if u.startswith(('http://','https://','#')):continue
  assert(p.parent/u.split('#')[0]).resolve().exists(),(p,u)
shutil.copy2('/tmp/validate-kv-owner-evidence.py',P/'validate-evidence.py')
files=subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(E)],cwd=R,text=True).splitlines();inv=[]
for rel in files:
 f=R/rel
 if f.name=='evidence-manifest.json':continue
 b=f.read_bytes();assert b[:4]!=b'\x7fELF',f
 if f.suffix=='.py':compile(b,str(f),'exec')
 if f.suffix=='.json':json.loads(b)
 inv.append({'path':str(f.relative_to(E)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
(P/'evidence-manifest.json').write_text(json.dumps({'complete':True,'files':inv,'proposal_sha256':hashlib.sha256(D.read_bytes()).hexdigest(),'proposal_status':'Review pending; no implementation.','note':'Raw JSONL200612169bytes and compiled binaries ignored; deterministic gzip/hash roundtrip validated in payload.json. Production remains unchanged.'},indent=2)+'\n')
subprocess.run(['git','diff','--check'],cwd=R,check=True)
assert hashlib.sha256((R/'build-aliceai/bin/libggml-base.so.0').read_bytes()).hexdigest()=='760e032ea9a7131dd3242ac2036ac8a12c8e7b05882d1dc40c39bebc9ec624bd'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
print('Verified',len(inv),'versionable files;',sum(x['bytes']for x in inv),'bytes; no ELF; links/scripts/JSON passed; production SHA unchanged; GPU free.')
