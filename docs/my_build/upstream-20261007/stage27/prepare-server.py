#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil
R=Path(__file__).resolve().parent;OLD=R.parent/'stage26';OUT=R/'experiment-bin'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old=json.loads((OLD/'build-manifest.json').read_text());candidate=json.loads((R/'build-manifest.json').read_text())
assert all(sha(OLD/'experiment-bin'/n)==h for n,h in old['experiment_hashes'].items())
assert not OUT.exists();shutil.copytree(OLD/'experiment-bin',OUT,symlinks=True)
shutil.copy2(R/'candidate-bin/libllama.so.0.4.0',OUT/'libllama.so.0.4.0')
hashes={n:sha(OUT/n) for n in old['experiment_hashes']}
assert {n for n,h in hashes.items() if h!=old['experiment_hashes'][n]}=={'libllama.so','libllama.so.0','libllama.so.0.4.0'}
manifest={'head':candidate['head'],'experiment_hashes':hashes,'parent_server_impl_sha256':old['experiment_hashes']['libllama-server-impl.so'],
          'source_sha256':candidate['source_sha256'],'server_source_sha256':old['sources']['server-context-capture.cpp'],
          'callback_sha256':sha(R/'general-callback.h'),'raw_layout_sha256':sha(R/'raw-layout.h'),
          'server_rebuilt':False,'libllama_rebuilt':True}
(R/'server-build-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
s=(OLD/'run-server-capture.py').read_text().replace("ROOT/'build-manifest.json'","ROOT/'server-build-manifest.json'")
s=s.replace("'production_change':False","'production_change':True")
s=s.replace(" profile=base.PROFILES['deepseek41'];", " assert sha(REPO/'src/llama-kv-cache.cpp')==json.loads((ROOT/'server-build-manifest.json').read_text())['source_sha256']\n profile=base.PROFILES['deepseek41'];")
(R/'run-server.py').write_text(s)
print('Existing capture server linked at runtime to the fixed isolated libllama')
