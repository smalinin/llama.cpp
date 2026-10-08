#!/usr/bin/env python3
"""Summarize the gated CUDA trace without equating summed time to wall time."""
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
db = sqlite3.connect(ROOT/'nsys-dspark.sqlite')
marker = db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='baseline-diagnostic-2'").fetchall()
assert len(marker)==1 and marker[0][1]>marker[0][0]
start,end = marker[0]
assert db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY').fetchone()[0]>0
assert db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]>0

copies = db.execute('''
    SELECT s.value,e.label,COUNT(*),SUM(m.bytes),SUM(m.end-m.start)/1e6
    FROM CUPTI_ACTIVITY_KIND_MEMCPY m
    JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId
    JOIN StringIds s ON s.id=r.nameId
    JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
    GROUP BY s.value,m.copyKind
    ORDER BY s.value,m.copyKind
''').fetchall()
matched = sum(row[2] for row in copies)
count = db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY').fetchone()[0]
assert matched==count
assert db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchone()[0] == \
       db.execute('SELECT COUNT(DISTINCT correlationId) FROM CUPTI_ACTIVITY_KIND_RUNTIME').fetchone()[0]

summary = {
    'capture':'baseline-diagnostic-2', 'includes_prefill':True,
    'nvtx_start_ns':start,'nvtx_end_ns':end,'nvtx_wall_ms':(end-start)/1e6,
    'copy_count':count,'kernel_count':db.execute('SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0],
    'copy_totals_by_api':[dict(zip(('api','direction','count','bytes','sum_gpu_ms'),row)) for row in copies],
    'top_kernels':[
        dict(zip(('name','count','sum_gpu_ms'),row)) for row in db.execute('''
        SELECT s.value,COUNT(*),SUM(k.end-k.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        GROUP BY k.shortName ORDER BY SUM(k.end-k.start) DESC LIMIT 15
    ''')],
    'runtime_memcpy_apis':[
        dict(zip(('api','count','sum_cpu_api_ms'),row)) for row in db.execute('''
        SELECT s.value,COUNT(*),SUM(r.end-r.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        WHERE s.value LIKE '%Memcpy%' GROUP BY s.value
    ''')],
    'limitations':[
        'Summed device activity can overlap; it is not serialized wall time.',
        'The captured request includes target prefill, target decode and DSpark work.',
        'There is no native CUDA trace in this stage; copy cost is not exclusively DSpark overhead.',
        'Peer-copy API is correlated with separate DtoH/HtoD activities in this trace; peer payload is counted twice across those directions.',
        'Kernel-level CUDA graph tracing adds instrumentation overhead.',
    ],
}
(ROOT/'nsys-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
