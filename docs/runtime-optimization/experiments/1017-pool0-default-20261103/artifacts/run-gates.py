from pathlib import Path
import json,time,subprocess

def main():
    P=Path(__file__).parent;R=Path('/home/sergei/Github/llama.cpp');started=time.monotonic()
    while True:
     try:j=json.load(open(P/'precision-results.json'))
     except json.JSONDecodeError:time.sleep(.2);continue
     assert not any(x.get('error')for x in j['arms']),j
     if j['complete']:break
     assert time.monotonic()-started<7200,'precision wait timeout';time.sleep(2)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU busy'
    for name in ['run-metadata.py','bench.py','admit.py','run-payload.py']:
     print('START',name,flush=True)
     with(P/(name+'.driver.log')).open('wb')as f:subprocess.run(['python3',str(P/name)],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
     print('DONE',name,flush=True)
    print('ALL_CURRENT_DEFAULT_GATES_DONE',flush=True)

if __name__ == '__main__':
    main()
