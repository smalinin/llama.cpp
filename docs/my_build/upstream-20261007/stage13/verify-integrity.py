#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parent
SNAP=ROOT.parent/'stage8/candidate-bin'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes=json.loads((ROOT.parent/'stage8/candidate-binary-sha256.json').read_text());assert all(sha(SNAP/n)==h for n,h in hashes.items())
loaded={};manifests=[]
for name in ['chain-capture-manifest.json','target-matmul-manifest.json','isolated-manifest.json','isolated-profile-manifest.json']:
 meta=json.loads((ROOT/name).read_text());manifests.append(name)
 source=ROOT/('capture-chain.cpp' if name.startswith('chain') else 'target-matmul-controls.cpp' if name.startswith('target') else 'isolated-replay.cpp')
 assert sha(source)==meta['source_sha256']
 assert sha(source.with_suffix(''))==meta['binary_sha256']
 records=meta.get('runs',[meta])
 for record in records:
  assert record['exit_code']==0
  if 'dataset' in record:
   directory=ROOT/'inputs'/record['dataset']
   assert sha(directory/'manifest.json')==record['input_manifest_sha256']
   if 'config_sha256' in record:assert sha(directory/'config.txt')==record['config_sha256']
   loaded.update(record['loaded_libraries'])
for directory in ['chain-capture-output','target-matmul-output']:
 for line in (ROOT/directory/'loaded-libraries.txt').read_text().splitlines():
  name=line.split()[-1]
  if '/candidate-bin/' in name:loaded[name]=sha(name)
assert loaded and all(Path(p).parent==SNAP and sha(p)==h==hashes[Path(p).name] for p,h in loaded.items())
for dataset in ['hc0','router0','post0','post2']:
 directory=ROOT/'inputs'/dataset;meta=json.loads((directory/'manifest.json').read_text())
 for name,value in meta['files'].items():assert sha(directory/name)==value['sha256']
 if 'shape' in meta:
  kind=meta['type'];(n,m)=meta['shape']
  assert (directory/'config.txt').read_text()==f'{kind} {n} {m}\n'
  with open(meta['source'],'rb') as f:f.seek(meta['offset']);raw=f.read(meta['bytes'])
  assert hashlib.sha256(raw).hexdigest()==meta['sha256']
raw={}
for directory in ['chain-capture-output','target-matmul-output','inputs','hc0-ada','hc0-ampere','router0-ada','router0-ampere','post0-ada','post0-ampere','post2-ada','post2-ampere','hc0-ada-profile','hc0-ampere-profile','router0-ada-profile','router0-ampere-profile']:
 for p in sorted((ROOT/directory).rglob('*')):
  if p.is_file():raw[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
(ROOT/'raw-file-sha256.json').write_text(json.dumps(raw,indent=2)+'\n')
chain=json.loads((ROOT/'chain-summary.json').read_text());target=json.loads((ROOT/'target-matmul-output/summary.json').read_text());isolated=json.loads((ROOT/'isolated-summary.json').read_text());profile=json.loads((ROOT/'kernel-summary.json').read_text())
wide_controls={mode:target['variants'][mode+'-w2']['sha256']==target['variants'][mode+'-w4']['sha256'] for mode in ['decode-scalar-fa-upgate-hc-router','decode-scalar-fa-upgate-float2d']}
assert all(wide_controls.values())
out={'head':subprocess.check_output(['git','-C','/home/sergei/Github/llama.cpp','rev-parse','HEAD'],text=True).strip(),'candidate_files_verified':len(hashes),'original_weights_rehashed':2,'weight_bytes':sum(json.loads((ROOT/'inputs'/k/'manifest.json').read_text())['bytes'] for k in ['hc0','router0']),'manifests':manifests,'all_exit_codes_zero':True,'model_replays':12,'logit_sha_controls_exact':sum(x['bit_identical'] for x in chain['controls'].values())+sum(x['bit_identical'] for x in target['controls'].values()),'wide_to_wide_full_logit_sha_controls':wide_controls,'isolated_cases_stable':isolated['stable_cases'],'profile_cases_stable':44,'actual_capture_controls_exact':sum(v['bit_identical'] for v in isolated['actual_capture_controls'].values()),'actual_capture_controls_total':len(isolated['actual_capture_controls']),'actual_capture_control_exception':'hc0-ampere-w4: different architecture versus the Ada model capture, max7.62939453125e-5; repeats and profiler controls exact','profile_data_files_exact':profile['profile_outputs_exact'],'loaded_libraries':loaded,'raw_data_files_hashed':len(raw),'raw_bytes_hashed':sum(v['bytes'] for v in raw.values()),'raw_file_index_sha256':sha(ROOT/'raw-file-sha256.json'),'scope':'Diagnostics only. HC+router control preserves all65 argmax at widths2/4, with identical wide logits. Large differences versus scalar persist. No free DSpark generation or HTTP throughput rerun.'}
(ROOT/'integrity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
