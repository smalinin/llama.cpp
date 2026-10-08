#!/usr/bin/env python3
"""Compare GGUF metadata, tokenizer encodings and tensor layouts without loading weights."""
import collections
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
FORMATS = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}

def u32(f):
    return struct.unpack('<I', f.read(4))[0]

def u64(f):
    return struct.unpack('<Q', f.read(8))[0]

def string(f):
    return f.read(u64(f)).decode('utf-8')

def value(f, kind):
    if kind in FORMATS:
        fmt = '<' + FORMATS[kind]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if kind == 8:
        return string(f)
    if kind == 9:
        element, count = u32(f), u64(f)
        start = f.tell()
        if count < 256:
            result = [value(f, element) for _ in range(count)]
        else:
            result = None
            if element in FORMATS:
                f.seek(count * struct.calcsize('<' + FORMATS[element]), 1)
            else:
                for _ in range(count): value(f, element)
        end = f.tell()
        f.seek(start)
        digest = hashlib.sha256(f.read(end-start)).hexdigest()
        return {'element_type':element, 'count':count, 'sha256':digest, 'values':result}
    raise ValueError(kind)

def inspect(path):
    with path.open('rb') as f:
        magic,version,count,keys = struct.unpack('<4sIQQ',f.read(24))
        assert magic == b'GGUF' and version == 3
        metadata = {}
        for _ in range(keys):
            key = string(f)
            metadata[key] = value(f,u32(f))
        tensors = []
        for _ in range(count):
            name = string(f)
            dims = [u64(f) for _ in range(u32(f))]
            kind,offset = u32(f),u64(f)
            tensors.append({'name':name, 'shape':dims, 'ggml_type':kind})
    return {'path':str(path), 'size':path.stat().st_size, 'mtime_ns':path.stat().st_mtime_ns,
            'metadata':metadata, 'tensors':tensors}

def main():
    old = json.loads((ROOT.parent/'stage0/model-metadata.json').read_text())
    paths = {item['role']:Path(item['path']) for item in old if item['profile']=='deepseek41'}
    models = {key:inspect(path) for key,path in paths.items()}
    target,draft = (models[key]['metadata'] for key in ('model','draft'))
    tokenizer = {key:{'target':target.get(key), 'draft':draft.get(key), 'equal':target.get(key)==draft.get(key)}
                 for key in sorted(set(target)|set(draft)) if key.startswith('tokenizer.') and key != 'tokenizer.chat_template'}
    counts = collections.Counter()
    for path in sorted(paths['model'].parent.glob('*.gguf')):
        shard = models['model'] if path==paths['model'] else inspect(path)
        counts.update(str(t['ggml_type']) for t in shard['tensors'])
    models['model']['all_shard_tensor_types'] = dict(counts)
    models['draft']['tensor_types'] = dict(collections.Counter(str(t['ggml_type']) for t in models['draft']['tensors']))
    alternative = []
    for path in (Path('/mdd/D2W/models/unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf'),
                 Path('/mdd/D2W/models/bartowski/DeepSeek-V4-Flash-0731-GGUF/dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf')):
        if path.exists(): alternative.append(inspect(path))
    result = {'models':models, 'tokenizer_comparison':tokenizer, 'other_local_drafts':alternative}
    (ROOT/'model-inspection.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Tokenizer token arrays equal:',tokenizer['tokenizer.ggml.tokens']['equal'])
    print('Target embedding length:',target.get('deepseek41.embedding_length'))
    print('Draft target layers:',draft.get('dflash.target_layers'))
    print('Draft flags:',{k:v for k,v in draft.items() if 'dflash.' in k and any(s in k for s in ('dsv41','block_size','anchor','confidence'))})
    print('Draft tensor types:',models['draft']['tensor_types'])
    print('Other draft names:',[(v['metadata'].get('general.name'),v['metadata'].get('dflash.dsv41_semantics')) for v in alternative])

if __name__ == '__main__': main()
