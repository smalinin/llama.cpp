#!/usr/bin/env python3
"""Check GLM-5.3 sampling repeatability using the saved Stage 5 binary."""

import argparse
import datetime
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
SERVER = ROOT.parent / 'stage5/candidate-bin/llama-server'
GPU_ORDER = json.loads((ROOT.parent / 'stage4/real-after/qwen4exp-spec-0/manifest.json').read_text())['gpu_order']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(mtp, launch, output):
    label = f'mtp{int(mtp)}-launch{launch}'
    directory = output / label
    directory.mkdir(parents=True, exist_ok=False)
    original = ROOT.parent / f'stage0/runs/glm-dsa-spec-{int(mtp)}'
    saved = json.loads((original / 'manifest.json').read_text())
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = saved['command'].copy()
    command[0] = str(SERVER)
    command[command.index('--port') + 1] = str(port)
    command += ['--verbosity', '5']
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('LLAMA_ARG_', 'LLAMA_MTP_', 'LLAMA_DSPARK_', 'GGML_SCHED_')):
            env.pop(key)
    for key in ('GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION',
                'LLAMA_FUSED_LID_DISABLE', 'QWEN4EXP_FUSED_LID', 'NVIDIA_TF32_OVERRIDE'):
        env.pop(key, None)
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=GPU_ORDER,
               LD_LIBRARY_PATH=str(SERVER.parent))
    hashes = json.loads((ROOT.parent / 'stage5/candidate-binary-sha256.json').read_text())
    for name, expected in hashes.items():
        if sha(SERVER.parent / name) != expected:
            raise RuntimeError('snapshot hash mismatch: ' + name)
    model = Path(saved['model'])
    manifest = {
        'command': command, 'mtp': mtp, 'launch': launch, 'gpu_order': GPU_ORDER,
        'cuda_device_order': 'PCI_BUS_ID', 'ld_library_path': env['LD_LIBRARY_PATH'],
        'date_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'tested_source_head': '11638b68545860e96b055798e995bb14be3d0e88',
        'tested_snapshot': 'stage5/candidate-bin', 'includes_pending_continue_fix': False,
        'model_files': [{'path': str(p), 'realpath': str(p.resolve()),
                         'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                        for p in sorted(model.parent.glob('GLM-5.3-UD-IQ3_XXS-*.gguf'))],
        'prompt_sha256': sha(original / 'prompt.txt'),
        'snapshot_hashes_verified': True,
    }
    if manifest['prompt_sha256'] != saved['prompt_sha256']:
        raise RuntimeError('saved prompt hash mismatch')
    base.save(directory / 'manifest.json', manifest)
    (directory / 'prompt.txt').write_text((original / 'prompt.txt').read_text())
    peaks = {'started': time.monotonic(), 'gpu_mib': {}, 'process_hwm_kib': 0}
    stop = threading.Event()
    result = {'case': label, 'status': 'running', 'requests': []}
    print('START', label, flush=True)
    with (directory / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        watcher = threading.Thread(target=base.monitor, args=(proc, directory, stop, peaks), daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic() + 1200
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f'server startup exit {proc.returncode}')
                try:
                    if base.api(port, '/health', timeout=3).get('status') == 'ok':
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('startup exceeded 1200 s')
                time.sleep(1)
            result['startup_s'] = time.monotonic() - peaks['started']
            maps = Path(f'/proc/{proc.pid}/maps').read_text().splitlines()
            libs = sorted({line.split()[-1] for line in maps if any('/' + prefix in line for prefix in
                           ('libllama', 'libggml', 'libmtmd'))})
            if not libs or any(Path(f).parent.resolve() != SERVER.parent.resolve() for f in libs):
                raise RuntimeError('wrong loaded libraries: ' + repr(libs))
            manifest['loaded_libraries'] = {f: sha(f) for f in libs}
            for f, value in manifest['loaded_libraries'].items():
                if hashes.get(Path(f).name) != value:
                    raise RuntimeError('loaded library hash mismatch: ' + f)
            base.save(directory / 'manifest.json', manifest)
            base.save(directory / 'props.json', base.api(port, '/props'))
            print('READY', label, round(result['startup_s'], 1), flush=True)
            modes = ['sampling', 'greedy'] if mtp else ['sampling']
            for mode in modes:
                payload = json.loads((original / f'{mode}-1-request.json').read_text())
                for repeat in range(1, 6):
                    name = f'{mode}-{repeat}'
                    base.save(directory / f'{name}-request.json', payload)
                    started = time.monotonic()
                    response = base.api(port, '/completion', payload, timeout=600)
                    base.save(directory / f'{name}-response.json', response)
                    if 'error' in response or not response.get('tokens'):
                        raise RuntimeError('completion failed: ' + repr(response))
                    if len(response['tokens']) != response.get('tokens_predicted'):
                        raise RuntimeError('incomplete token IDs')
                    if response['timings'].get('cache_n', 0) != 0:
                        raise RuntimeError('unexpected prompt reuse')
                    item = {'request': name, 'wall_s': time.monotonic() - started,
                            'timings': response['timings'], 'tokens_predicted': response['tokens_predicted'],
                            'content_sha256': hashlib.sha256(response['content'].encode()).hexdigest(),
                            'tokens_sha256': hashlib.sha256(json.dumps(response['tokens']).encode()).hexdigest()}
                    result['requests'].append(item)
                    base.save(directory / 'result.json', result)
                    print('DONE', label, name, json.dumps(item['timings']), flush=True)
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
            text = (directory / 'server.log').read_text(errors='replace')
            assignments = re.findall(r'load_tensors: layer\s+(\d+) assigned to device ([^\n]+)', text)
            manifest['layer_placement'] = {index: value for index, value in assignments}
            manifest['placement_note'] = 'Last assignment per layer; full fit attempts remain in server.log.'
            manifest['model_buffers'] = re.findall(r'load_tensors:\s+[^\n]*model buffer size =[^\n]+', text)
            manifest['compute_buffers'] = re.findall(r'[^\n]*compute buffer size =[^\n]+', text)
            manifest['kv_buffers'] = re.findall(r'[^\n]*KV buffer size =[^\n]+', text)
            base.save(directory / 'manifest.json', manifest)
            result.update(server_exit_code=proc.returncode, peak_gpu_used_mib=peaks['gpu_mib'],
                          peak_process_rss_kib=peaks['process_hwm_kib'], total_s=time.monotonic() - peaks['started'])
            base.save(directory / 'result.json', result)
    print('FINISH', label, result['status'], result.get('error', ''), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
                                   '--format=csv'], text=True)
    (args.output / 'gpus.csv').write_text(gpu)
    results = []
    for mtp, launch in [(True, 1), (True, 2), (False, 1), (False, 2)]:
        results.append(run(mtp, launch, args.output))
        base.save(args.output / 'results.json', results)


if __name__ == '__main__':
    main()
