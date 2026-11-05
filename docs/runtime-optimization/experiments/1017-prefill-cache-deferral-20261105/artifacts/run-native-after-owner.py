from pathlib import Path
import subprocess,json,time,shutil
R=Path('/home/sergei/Github/llama.cpp');P=R/'docs/runtime-optimization/experiments/1017-prefill-cache-deferral-20261105/artifacts';D=P/'owner-on';start=time.monotonic()
while True:
    if (D/'counts-results.json').exists():
        j=json.loads((D/'counts-results.json').read_text())
        if j.get('error')or j.get('floor_breach')or j.get('warnings'):raise RuntimeError('Owner diagnostic failed; inspect its original manifest.')
        if j.get('complete'):break
        if j.get('exit_code')not in (None,0):raise RuntimeError('Owner diagnostic exited with error.')
    if time.monotonic()-start>1800:raise RuntimeError('Owner diagnostic did not finish within30min.')
    time.sleep(1)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with(D/'analysis.log').open('wb')as log:
    result=subprocess.run(['python3',str(D/'analyze.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Actual owner/flush audit failed; no native launch.'
owner=json.loads((D/'owner-checks.json').read_text());assert owner['complete'];print('OWNER_GATE_COMPLETE',owner['actual_origin_owners'],flush=True)
with(P/'native-run.log').open('wb')as log:
    result=subprocess.run(['python3',str(P/'native.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Native comparison failed; preserve manifests.'
shutil.copy2('/tmp/run-deferral-native-after-owner.py',P/'run-native-after-owner.py');print('NATIVE_GATE_COMPLETE',flush=True)
