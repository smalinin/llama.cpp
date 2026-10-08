#!/usr/bin/env python3
"""Compare Qwen QSA on fixed GPUs, context sizes, slots and MTP modes."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/sergei/Github/llama.cpp')
spec = importlib.util.spec_from_file_location('baseline', ROOT.parent / 'run_baseline.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
GPU_ORDER = ','.join(json.loads((ROOT.parent / 'stage4/real-after/qwen4exp-spec-0/manifest.json').read_text())['gpu_order'].split(','))


def prompt(count):
    return 'Read the following numbered facts.\n' + ''.join(
        f'Entry {i}: the checkpoint value is {1000+i}; the label is blue.\n' for i in range(count)
    ) + '\nExplain how to check these entries, then list the first ten checkpoint values.\nAnswer:\n'


def run(version, context, slots, mtp):
    label = f'c{context}-np{slots}-mtp{int(mtp)}'
    directory = ROOT / ('real-' + version) / label
    directory.mkdir(parents=True, exist_ok=False)
    server = ROOT / (version + '-bin/llama-server')
    profile = base.PROFILES['qwen4exp']
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    command = [str(server), '-m', str(profile['model']), '-c', str(context), '-b', '2048', '-ub', '512',
               '-np', str(slots), '-ctk', 'f16', '-ctv', 'f16', '-ctkd', 'f16', '-ctvd', 'f16',
               '-t', '12', '-tb', '12', '-fit', 'off', '-ngl', '99', '--tensor-split', '1,1,1,1,1,0.4',
               '--split-mode', 'layer', '--load-mode', 'none', '--lazy-mode', 'auto',
               '--host', '127.0.0.1', '--port', str(port), '--metrics', '--verbosity', '5']
    if mtp:
        command += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '--model-draft', str(profile['draft'])]
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('LLAMA_ARG_', 'LLAMA_MTP_', 'LLAMA_DSPARK_', 'GGML_SCHED_')):
            env.pop(key)
    for key in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION', 'LLAMA_FUSED_LID_DISABLE', 'QWEN4EXP_FUSED_LID'):
        env.pop(key, None)
    if version in ('after', 'review', 'candidate'):
        env['QWEN4EXP_FUSED_LID'] = '1'
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=GPU_ORDER, LD_LIBRARY_PATH=str(server.parent))
    manifest = {'command': command, 'gpu_order': GPU_ORDER, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'base_head': (ROOT / 'base-head.txt').read_text().strip(), 'context': context, 'slots': slots, 'mtp': mtp,
                'qwen_fused_lid_requested': env.get('QWEN4EXP_FUSED_LID', 'unset')}
    base.save(directory / 'manifest.json', manifest)
    peaks = {'started': time.monotonic(), 'gpu_mib': {}, 'process_hwm_kib': 0}
    stop = threading.Event()
    result = {'case': label, 'version': version, 'status': 'running', 'requests': []}
    print('START', version, label, flush=True)
    with (directory / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        watcher = threading.Thread(target=base.monitor, args=(proc, directory, stop, peaks), daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server startup exit {proc.returncode}')
                try:
                    if base.api(port, '/health', timeout=3).get('status') == 'ok':
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('startup exceeded 900 s')
                time.sleep(1)
            result['startup_s'] = time.monotonic() - peaks['started']
            maps = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            libs = sorted({line.split()[-1] for line in maps if any('/' + prefix in line for prefix in
                           ('libllama.so.', 'libllama-server-impl.so', 'libggml-cuda.so.', 'libggml-base.so.'))})
            if len(libs) != 4 or any(Path(f).parent.resolve() != server.parent.resolve() for f in libs):
                raise RuntimeError('wrong loaded libraries: ' + repr(libs))
            manifest['loaded_libraries'] = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in libs}
            base.save(directory / 'props.json', base.api(port, '/props'))
            phases = [('short', prompt(80))]
            if context > 8192:
                phases += [('long', prompt(600))]
            for phase, text in phases:
                (directory / (phase + '-prompt.txt')).write_text(text)
                for repeat in range(1, 3):
                    def request(slot):
                        label = f'{phase}-{repeat}-slot{slot}'
                        payload = {'prompt': text, 'n_predict': 64, 'temperature': 0, 'seed': 1234,
                                   'cache_prompt': False, 'return_tokens': True, 'stream': False, 'id_slot': slot}
                        base.save(directory / (label + '-request.json'), payload)
                        response = base.api(port, '/completion', payload, timeout=600)
                        base.save(directory / (label + '-response.json'), response)
                        if 'error' in response or not response.get('tokens'):
                            raise RuntimeError('completion failed: ' + repr(response))
                        return {'request': label, 'timings': response['timings']}
                    with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as pool:
                        result['requests'] += list(pool.map(request, range(slots)))
                    base.save(directory / 'result.json', result)
                    print('DONE', version, result['case'], phase, repeat, flush=True)
            text = (directory / 'server.log').read_text(errors='replace')
            manifest['layer_placement'] = re.findall(r'load_tensors: layer\s+\d+ assigned to device [^\n]+', text)
            manifest['model_buffers'] = re.findall(r'load_tensors:\s+[^\n]*model buffer size =[^\n]+', text)
            manifest['compute_buffers'] = re.findall(r'[^\n]*compute buffer size =[^\n]+', text)
            manifest['qsa_gather_events'] = re.findall(r'[^\n]*(?:gather|Lightning Indexer)[^\n]*', text)
            if not manifest['layer_placement'] or not manifest['compute_buffers']:
                raise RuntimeError('missing placement or buffer evidence')
            base.save(directory / 'manifest.json', manifest)
            result['status'] = 'passed'
        except Exception as error:
            result.update(status='failed', error=repr(error))
        finally:
            stop.set()
            watcher.join(timeout=15)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            result.update(server_exit_code=proc.returncode, peak_gpu_used_mib=peaks['gpu_mib'],
                          peak_process_rss_kib=peaks['process_hwm_kib'], total_s=time.monotonic() - peaks['started'])
            base.save(directory / 'result.json', result)
    print('FINISH', version, result['case'], result['status'], result.get('error', ''), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError(result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', choices=['before', 'after', 'final', 'review', 'candidate'])
    parser.add_argument('--contexts', nargs='+', type=int, default=[8192, 65536])
    parser.add_argument('--slots', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--mtp', nargs='+', type=int, choices=[0, 1], default=[0, 1])
    args = parser.parse_args()
    results = []
    for context in args.contexts:
        for slots in args.slots:
            for mtp in map(bool, args.mtp):
                results.append(run(args.version, context, slots, mtp))
                base.save(ROOT / ('real-' + args.version + '-summary.json'), results)
