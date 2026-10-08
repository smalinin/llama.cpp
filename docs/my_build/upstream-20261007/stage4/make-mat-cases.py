#!/usr/bin/env python3
from pathlib import Path
import ctypes,itertools
R=Path(__file__).resolve().parent
lib=ctypes.CDLL(str(R/'before-bin/libggml-base.so'));lib.ggml_op_name.restype=ctypes.c_char_p;lib.ggml_type_name.restype=ctypes.c_char_p
op=next(i for i in range(100) if lib.ggml_op_name(i)==b'MUL_MAT')
types={t:next(i for i in range(40) if lib.ggml_type_name(i)==t.encode()) for t in ('f32','f16','bf16')}
def line(dtype,k,m,n,heads,layout,name):
    if layout=='pad':anb=[2,2*(k+4),2*(k+4)*m,2*(k+4)*m*heads]
    elif layout=='permute':anb=[2,2*k*heads,2*k,2*k*m*heads]
    else:anb=[2,2*k,2*k*m,2*k*m*heads]
    bnb=[4,4*k,4*k*n,4*k*n*heads]
    return ' '.join(map(str,[op,types['f32'],m,n,heads,1,0,2,types[dtype],k,m,heads,1,*anb,types['f32'],k,n,heads,1,*bnb,name]))
perf=[]
for dtype,m,n in itertools.product(('f16','bf16'),(32,33),(4,32)):
    perf.append(line(dtype,128,m,n,1,'contiguous',f'{dtype}-m{m}-n{n}'))
(R/'mat-perf-cases.txt').write_text('\n'.join(perf)+'\n')
print('perf cases:',len(perf))
