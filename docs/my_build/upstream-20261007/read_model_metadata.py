#!/usr/bin/env python3
"""Read selected GGUF metadata with the Python standard library."""

import json
from pathlib import Path
import struct
import sys


FORMATS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def read_value(stream, kind, keep=False):
    if kind in FORMATS:
        fmt = '<' + FORMATS[kind]
        return struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]
    if kind == 8:
        length = struct.unpack('<Q', stream.read(8))[0]
        if keep:
            return stream.read(length).decode('utf-8')
        stream.seek(length, 1)
        return None
    if kind == 9:
        element, length = struct.unpack('<IQ', stream.read(12))
        if element in FORMATS:
            stream.seek(length * struct.calcsize('<' + FORMATS[element]), 1)
        else:
            for _ in range(length):
                read_value(stream, element)
        return {'element_type': element, 'count': length} if keep else None
    raise ValueError(f'Unknown GGUF type {kind}')


def inspect(path):
    with path.open('rb') as stream:
        magic, version, _, count = struct.unpack('<4sIQQ', stream.read(24))
        if magic != b'GGUF' or version not in (2, 3):
            raise ValueError(f'Unsupported GGUF header: {path}')
        fields = {}
        for _ in range(count):
            key = read_value(stream, 8, True)
            kind = struct.unpack('<I', stream.read(4))[0]
            keep = key in ('general.architecture', 'general.name', 'split.count') or any(
                part in key for part in ('context_length', 'block_count', 'nextn', 'expert_count', 'dspark', 'dflash', 'block_size')
            )
            value = read_value(stream, kind, keep)
            if keep:
                fields[key] = value
    shards = list(path.parent.glob(path.name.split('-00001-of-')[0] + '-*-of-*.gguf')) if '-00001-of-' in path.name else [path]
    return {
        'path': str(path), 'resolved': str(path.resolve()), 'shards': len(shards),
        'total_bytes': sum(item.stat().st_size for item in shards), 'gguf_version': version, 'fields': fields,
    }


if __name__ == '__main__':
    print(json.dumps([inspect(Path(arg)) for arg in sys.argv[1:]], ensure_ascii=False, indent=2))
