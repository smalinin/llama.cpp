#!/usr/bin/env python3
import struct,json,sys
from pathlib import Path
selected=[]
for model in sys.argv[1:]:
    f=Path(model).open('rb')
    def get(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
    def string():return f.read(get('Q')).decode('utf-8',errors='replace')
    sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
    def skip(typ):
        if typ in sizes:f.seek(sizes[typ],1)
        elif typ==8:f.seek(get('Q'),1)
        elif typ==9:
            sub=get('I');n=get('Q')
            if sub in sizes:f.seek(sizes[sub]*n,1)
            else:
                for _ in range(n):skip(sub)
        else:raise ValueError(typ)
    assert f.read(4)==b'GGUF';version=get('I');n=get('Q');kv=get('Q')
    for _ in range(kv):string();skip(get('I'))
    for _ in range(n):
        name=string();d=get('I');shape=[get('Q') for _ in range(d)];typ=get('I');offset=get('Q')
        selected.append({'model':model,'name':name,'shape':shape,'ggml_type':typ})
    f.close()
print(json.dumps(selected,indent=2))
