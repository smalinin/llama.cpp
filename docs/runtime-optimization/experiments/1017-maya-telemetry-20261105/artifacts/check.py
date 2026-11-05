from pathlib import Path
import json,hashlib,subprocess,re,xml.etree.ElementTree as ET,gzip
R=Path('/home/sergei/Github/llama.cpp');E=R/'docs/runtime-optimization/experiments/1017-maya-telemetry-20261105';P=E/'artifacts';entries=[]
# Summarize locally retained heavy traces as well as reviewable sources/results.
for f in sorted(E.rglob('*')):
 if not f.is_file()or f.name in ['SHA256.json','review-checks.json']:continue
 h=hashlib.sha256()
 with f.open('rb')as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 entries.append({'relative_path':str(f.relative_to(E)),'absolute_locator':str(f),'bytes':f.stat().st_size,'sha256':h.hexdigest(),'local_heavy_or_compiled':f.suffix in ['.sqlite','.nsys-rep','.qdstrm','.so']or f.name=='probe.jsonl'})
(P/'SHA256.json').write_text(json.dumps({'scope':'Persistent diagnostic evidence. Heavy traces/compiled helpers/uncompressed stacks are local-only; no weights included. SHA256.json and review-checks.json are excluded to avoid self-reference.','entries':entries},indent=2)+'\n')
errors=[]
for match in re.finditer(r'\]\(([^)]+)\)',(E/'REPORT.md').read_text()):
 target=match.group(1)
 if not target.startswith(('http','/')) and not(E/target.split('#')[0]).exists():errors.append(target)
ET.parse(P/'prefill8k-timeline.svg')
for e in ['maya','ours']:
 f=P/e/'probe.jsonl';assert gzip.open(str(f)+'.gz','rb').read()==f.read_bytes()
review_files=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',str(E.relative_to(R))],cwd=R,text=True).splitlines();heavy_visible=[f for f in review_files if f.endswith(('.sqlite','.nsys-rep','.qdstrm','.so','probe.jsonl'))];assert not errors,errors;assert not heavy_visible,heavy_visible
checks={'report_links_valid':True,'SVG_XML_valid':True,'compressed_stacks_exact':True,'reviewable_files':len(review_files),'reviewable_bytes':sum((R/f).stat().st_size for f in review_files),'heavy_traces_excluded_from_git':True,'runtime_source_diff':subprocess.check_output(['git','diff','--','src','ggml/src','common','tools/server'],cwd=R,text=True),'original_Maya_controller_remains_incomplete':not json.loads((P/'results.json').read_text())['complete'],'ours_controller_complete':json.loads((P/'ours-results.json').read_text())['complete']}
assert checks['runtime_source_diff']==''
(P/'review-checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks));print('SHA files',len(entries),'local GB',sum(x['bytes']for x in entries)/1e9)
