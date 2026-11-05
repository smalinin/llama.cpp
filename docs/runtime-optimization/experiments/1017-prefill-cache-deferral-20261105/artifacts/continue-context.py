from pathlib import Path
import json,shutil
s=Path('/tmp/sweep-deferral-context.py').read_text().replace("for ctx in [99328,98304,94208]:","for ctx in [92160,90112,88064]:").replace("out={'complete':False,'tries':[]}","out=json.loads((P/'context-sweep.json').read_text());out['complete']=False")
s=s.replace("shutil.copy2('/tmp/sweep-deferral-context.py',P/'sweep-context.py')","shutil.copy2('/tmp/continue-deferral-context.py',P/'continue-context.py')")
Path('/tmp/continue-deferral-context-run.py').write_text(s)
