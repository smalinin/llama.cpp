#!/usr/bin/env python3
"""Capture comparable local llama-server baselines without changing source files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


REPO = Path('/home/sergei/Github/llama.cpp')
MODEL_ROOT = Path('/home/sergei/.models/unsloth')
PROFILES = {
    'glm5next': {
        'model': MODEL_ROOT / 'GLM-5.3-Flash-GGUF/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf',
    },
    'qwen4exp': {
        'model': MODEL_ROOT / 'Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf',
        'draft': MODEL_ROOT / 'Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf',
    },
    'glm-dsa': {
        'model': MODEL_ROOT / 'GLM-5.3-GGUF/GLM-5.3-UD-IQ3_XXS-00001-of-00007.gguf',
    },
    'deepseek41': {
        'model': Path('/media/sergei/WW2T/models/smalinin/DeepSeek-V4.1-Flash-New/DeepSeek-V4.1-Flash-Protected-Q2_K-Q8_0_Engram-imatrix/DeepSeek-V4.1-Flash-Protected-Q2_K-Q8_0_Engram-imatrix-00001-of-00010.gguf'),
        'draft': Path('/media/sergei/WW2T/models/smalinin/DeepSeek-V4.1-Flash/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf'),
        'spec_type': 'draft-dspark',
    },
}
GPU_ORDER = '2,1,0,5,4,3'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPT = (
    'Read the following numbered facts.\n'
    + ''.join(f'Entry {i}: the checkpoint value is {1000 + i}; the label is blue.\n' for i in range(96))
    + '\nExplain how to check these entries, then list the first ten checkpoint values.\nAnswer:\n'
)


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def api(port, path, payload=None, timeout=300):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}', data=body,
        headers={'Content-Type': 'application/json'},
    )
    with OPENER.open(req, timeout=timeout) as response:
        return json.load(response)


def monitor(proc, directory, stop, peaks):
    with (directory / 'memory.jsonl').open('w') as output:
        while not stop.is_set():
            sample = {'elapsed_s': time.monotonic() - peaks['started'], 'gpu_mib': {}}
            try:
                for line in Path(f'/proc/{proc.pid}/status').read_text().splitlines():
                    if line.startswith(('VmRSS:', 'VmHWM:')):
                        key, value, _ = line.split()
                        sample[key.rstrip(':') + '_kib'] = int(value)
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode:
                    sample['gpu_error'] = result.stderr.strip()
                for line in result.stdout.splitlines():
                    index, used = (int(item.strip()) for item in line.split(','))
                    sample['gpu_mib'][str(index)] = used
                    peaks['gpu_mib'][str(index)] = max(peaks['gpu_mib'].get(str(index), 0), used)
                peaks['process_hwm_kib'] = max(peaks['process_hwm_kib'], sample.get('VmHWM_kib', 0))
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                sample['monitor_error'] = str(error)
            output.write(json.dumps(sample) + '\n')
            output.flush()
            stop.wait(1)


def run_case(name, mtp, out, server):
    directory = out / f'{name}-spec-{int(mtp)}'
    directory.mkdir(exist_ok=False)
    profile = PROFILES[name]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [
        str(server), '-m', str(profile['model']),
        '-c', '8192', '-b', '2048', '-ub', '512', '-np', '1',
        '-ctk', 'f16', '-ctv', 'f16', '-ctkd', 'f16', '-ctvd', 'f16',
        '-t', '12', '-tb', '12', '-fit', 'on', '--fit-target', '3072',
        '--split-mode', 'layer', '--load-mode', 'none', '--lazy-mode', 'auto',
        '--host', '127.0.0.1', '--port', str(port), '--metrics',
    ]
    if mtp:
        command += ['--spec-type', profile.get('spec_type', 'draft-mtp'), '--spec-draft-n-max', '3']
        if 'draft' in profile:
            command += ['--model-draft', str(profile['draft'])]
    env = os.environ.copy()
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = GPU_ORDER
    for key in list(env):
        if key.startswith(('LLAMA_ARG_', 'LLAMA_MTP_', 'LLAMA_DSPARK_', 'GGML_SCHED_')):
            env.pop(key)
    manifest = {
        'command': command, 'gpu_order': GPU_ORDER, 'cuda_device_order': 'PCI_BUS_ID',
        'model': str(profile['model']),
        'speculative': mtp, 'spec_type': profile.get('spec_type', 'draft-mtp') if mtp else 'none',
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
    }
    save(directory / 'manifest.json', manifest)
    (directory / 'prompt.txt').write_text(PROMPT)
    peaks = {'started': time.monotonic(), 'gpu_mib': {}, 'process_hwm_kib': 0}
    stop = threading.Event()
    result = {'case': directory.name, 'status': 'running', 'requests': []}
    print(f'START {directory.name}', flush=True)
    with (directory / 'server.log').open('w') as log:
        proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        watcher = threading.Thread(target=monitor, args=(proc, directory, stop, peaks), daemon=True)
        watcher.start()
        try:
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f'server exited during startup: {proc.returncode}')
                try:
                    health = api(port, '/health', timeout=3)
                    if health.get('status') == 'ok':
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('server startup exceeded 900 seconds')
            result['startup_s'] = time.monotonic() - peaks['started']
            save(directory / 'props.json', api(port, '/props'))
            for request_name, temperature in [('greedy-1', 0), ('greedy-2', 0), ('sampling-1', 0.8), ('sampling-2', 0.8)]:
                payload = {
                    'prompt': PROMPT, 'n_predict': 128, 'temperature': temperature,
                    'seed': 1234, 'top_k': 40, 'top_p': 0.95, 'min_p': 0.05,
                    'cache_prompt': False, 'return_tokens': True, 'stream': False,
                }
                save(directory / f'{request_name}-request.json', payload)
                started = time.monotonic()
                response = api(port, '/completion', payload)
                save(directory / f'{request_name}-response.json', response)
                if 'error' in response:
                    raise RuntimeError(str(response['error']))
                item = {
                    'request': request_name, 'wall_s': time.monotonic() - started,
                    'timings': response.get('timings'), 'tokens_predicted': response.get('tokens_predicted'),
                    'content_sha256': hashlib.sha256(response.get('content', '').encode()).hexdigest(),
                    'tokens_sha256': hashlib.sha256(json.dumps(response.get('tokens')).encode()).hexdigest(),
                }
                result['requests'].append(item)
                save(directory / 'result.json', result)
                print(f'DONE {directory.name} {request_name}: {json.dumps(item["timings"])}', flush=True)
            result['status'] = 'passed'
            for mode in ('greedy', 'sampling'):
                a = json.loads((directory / f'{mode}-1-response.json').read_text())
                b = json.loads((directory / f'{mode}-2-response.json').read_text())
                result[f'{mode}_repeat_identical'] = a.get('content') == b.get('content') and a.get('tokens') == b.get('tokens')
        except Exception as error:
            result['status'] = 'failed'
            result['error'] = repr(error)
            print(f'FAIL {directory.name}: {error}', flush=True)
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
            result['server_exit_code'] = proc.returncode
            result['peak_gpu_used_mib'] = peaks['gpu_mib']
            result['peak_process_rss_kib'] = peaks['process_hwm_kib']
            result['total_s'] = time.monotonic() - peaks['started']
            save(directory / 'result.json', result)
    print(f'FINISH {directory.name}: {result["status"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', type=Path, default=REPO / 'build-glm53/bin/llama-server')
    parser.add_argument('--models', nargs='+', choices=PROFILES, default=list(PROFILES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in args.models:
        for mtp in (False, True):
            results.append(run_case(name, mtp, args.output, args.server))
            save(args.output / 'summary.json', results)
    raise SystemExit(0 if all(item['status'] == 'passed' for item in results) else 1)


if __name__ == '__main__':
    main()
