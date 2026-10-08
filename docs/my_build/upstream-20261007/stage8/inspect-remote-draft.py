#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
repository = 'Lucebox/DeepSeek-V4.1-Flash-DSpark-GGUF'
filename = 'DeepSeek-V4.1-Flash-DSpark-draft-Q2K-Q4K.gguf'
with urllib.request.urlopen('https://huggingface.co/api/models/' + repository, timeout=30) as response:
    info = json.load(response)
revision = info['sha']
url = 'https://huggingface.co/' + repository + '/resolve/' + revision + '/' + filename
request = urllib.request.Request(url, headers={'Range': 'bytes=0-8388607'})
with urllib.request.urlopen(request, timeout=60) as response:
    status = response.status
    content_range = response.headers.get('Content-Range')
    prefix = response.read(8 * 1024 * 1024)
path = ROOT / 'lucebox-q2k-q4k-header.gguf.part'
path.write_bytes(prefix)
spec = importlib.util.spec_from_file_location('inspection', ROOT.parent / 'stage7/inspect-models.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
data = module.inspect(path)
data.update(repository=repository, revision=revision, filename=filename,
            http_status=status, content_range=content_range, prefix_only=True)
(ROOT / 'lucebox-q2k-q4k-inspection.json').write_text(json.dumps(data, indent=2) + '\n')
print(json.dumps({'revision': revision, 'status': status, 'content_range': content_range,
                  'metadata': {k: v for k, v in data['metadata'].items() if not k.startswith('tokenizer.')},
                  'tensors': data['tensors']}, indent=2))
