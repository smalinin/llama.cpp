from pathlib import Path
import subprocess,json
R=Path(__file__).resolve().parent
cases=[['qwen4exp','qwen-final-v2','--faults','--restart','--q8'],['glm5next','glm5next-final','--restart'],['glm-dsa','glm-dsa-final','--q8'],['deepseek41','deepseek-final','--faults','--restart'],['tiny','native-final','--native','--restart']]
results=[]
for case in cases:
 print('START',case,flush=True)
 with (R/(case[1]+'.log')).open('w') as log:r=subprocess.run(['python3',str(R/'run-slot.py'),*case],stdout=log,stderr=subprocess.STDOUT)
 results.append({'case':case,'exit_code':r.returncode});(R/'slot-runs.json').write_text(json.dumps(results,indent=2)+'\n')
 print('DONE',case[1],r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
