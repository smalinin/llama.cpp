from pathlib import Path
import json,hashlib,gzip,re,subprocess,ast
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105';P=E/'artifacts';sha=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
assert json.loads((P/'native-results.json').read_text())['complete'];assert json.loads((P/'precision-results.json').read_text())['all_groups_exact'];assert json.loads((P/'owner-on/owner-checks.json').read_text())['complete'];assert json.loads((P/'all-cache-writes-checks.json').read_text())['complete'];fixture=json.loads((P/'scheduler-results.json').read_text());assert fixture['complete']and len(fixture['cases'])==42
parsed=[]
for f in P.rglob('*.json'):json.loads(f.read_text());parsed.append(str(f.relative_to(E)))
py=[]
for f in P.rglob('*.py'):ast.parse(f.read_text(),filename=str(f));py.append(str(f.relative_to(E)))
roundtrips=[]
for name in ['owner-default','owner-on']:
 d=P/name;j=json.loads((d/'payload.json').read_text());assert sha(d/'scopes.jsonl.gz')==j['gzip_sha256'];h=hashlib.sha256();size=0
 with gzip.open(d/'scopes.jsonl.gz','rb')as f:
  while b:=f.read(1024*1024):h.update(b);size+=len(b)
 assert h.hexdigest()==j['raw_sha256']and size==j['raw_bytes'];roundtrips.append(name)
b=json.loads((P/'build.json').read_text());assert sha(P/'candidate/libllama.so.0')==b['private_llama_sha256'];assert sha(R/'build-aliceai/bin/libllama.so.0')==b['production_llama_sha256'];assert sha(R/'build-aliceai/bin/libggml-base.so.0')==b['production_base_sha256'];assert subprocess.check_output(['git','diff','--','src'],cwd=R,text=True)==(P/'runtime.patch').read_text()
for src,final in [('src/models/deepseek41.cpp','deepseek41.final.cpp'),('src/llama-kv-cache-dsv4.cpp','llama-kv-cache-dsv4.final.cpp'),('src/llama-kv-cache-dsv4.h','llama-kv-cache-dsv4.final.h')]:assert sha(R/src)==sha(P/final)
links=[]
for doc in [E/'REPORT.md',R/'docs/runtime-optimization/designs/1017-prefill-cache-deferral.md']:
 for target in re.findall(r'\]\(([^)]+)\)',doc.read_text()):
  if '://'in target or target.startswith('#'):continue
  dest=(doc.parent/target.split('#')[0]);assert dest.exists(),(doc,target);links.append(target)
files=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z','--',str(E.relative_to(R))],cwd=R).split(b'\0');manifest=[]
for name in files:
 if not name:continue
 f=R/name.decode();head=f.open('rb').read(4);assert head!=b'\x7fELF',str(f);manifest.append({'path':str(f.relative_to(E)),'bytes':f.stat().st_size,'sha256':sha(f)})
subprocess.run(['git','diff','--check'],cwd=R,check=True);apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not apps,apps
j={'complete':True,'json_files_parsed':len(parsed),'python_sources_parsed':len(py),'owner_gzip_roundtrips':roundtrips,'fixture_cases':42,'native_requests':27,'precision_paired_groups':12,'all_local_report_design_links_exist':len(links),'canonical_source_patch_snapshots_match':True,'private_production_library_hashes_match_build_manifest':True,'no_versionable_elf':True,'git_diff_check':True,'gpu_processes_remaining':[],'versionable_payload_bytes':sum(x['bytes']for x in manifest),'files':manifest,'note':'Manifest lists the evidence payload before this self-describing final-check file. Ignored private modules/raw jsonl and retained local11GB initial observer trace have separate build/payload/cancellation locators; they are not versionable evidence payload.'};(P/'final-checks.json').write_text(json.dumps(j,indent=2)+'\n');print({k:v for k,v in j.items()if k!='files'})
