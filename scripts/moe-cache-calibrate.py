#!/usr/bin/env python3
"""Measure DeepSeek41 expert lanes and cache diagnostic costs; do not change runtime policy."""
import argparse
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import tempfile

FORMAT = "llama-moe-cost-calibration"
VERSION = 1
CASES = [(1, 1), (1, 6), (6, 6), (32, 6), (512, 6)]
MODES = ("gpu_resident", "cpu_lane", "serial_h2d_gpu")


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command(args, **kwargs):
    return subprocess.check_output(args, text=True, **kwargs).strip()


def gpus():
    keys = ["index", "uuid", "pci.bus_id", "name", "driver_version", "memory.total", "pcie.link.gen.max", "pcie.link.width.current"]
    rows = command(["nvidia-smi", "--query-gpu=" + ",".join(keys), "--format=csv,noheader,nounits"])
    return [dict(zip(keys, [x.strip() for x in row.split(",")])) for row in rows.splitlines()]


def file_identity(path):
    path = path.resolve(strict=True)
    st = path.stat()
    return {"path": str(path), "device": st.st_dev, "inode": st.st_ino, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def model_tensors(first, layers):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
    from gguf import GGUFReader

    first = first.absolute()
    match = re.fullmatch(r"(.*)-00001-of-(\d{5})[.]gguf", first.name)
    files = [first]
    if match:
        count = int(match[2])
        files = [first.with_name(f"{match[1]}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]
        files = [files[0], files[-1], *files[1:-1]] if count > 1 else files
    wanted = {f"blk.{layer}.ffn_{role}_exps.weight" for layer in layers for role in ("up", "gate", "down")}
    found = {}
    arch = None
    for path in files:
        reader = GGUFReader(path)
        if arch is None:
            arch = str(reader.get_field("general.architecture").contents())
            require(arch == "deepseek41", "only DeepSeek41 expert calibration is supported")
            declared_count = reader.get_field("split.count")
            require(declared_count is None or declared_count.contents() == len(files), "GGUF split count mismatch")
        for tensor in reader.tensors:
            if tensor.name not in wanted:
                continue
            require(tensor.name not in found, "duplicate tensor: " + tensor.name)
            shape = list(map(int, tensor.shape))
            require(len(shape) == 3, "expected 3D expert weights")
            info = file_identity(path)
            blockdev = Path(f"/sys/dev/block/{os.major(info['device'])}:{os.minor(info['device'])}")
            info["block_device"] = str(blockdev.resolve()) if blockdev.exists() else None
            with path.open("rb") as source:
                samples = []
                for position in (tensor.data_offset, tensor.data_offset + tensor.n_bytes - 128):
                    source.seek(position)
                    data = source.read(128)
                    require(len(data) == 128, "truncated tensor source")
                    samples.append(hashlib.sha256(data).hexdigest())
            found[tensor.name] = {"name": tensor.name, "path": info["path"], "offset": int(tensor.data_offset),
                                  "bytes": int(tensor.n_bytes), "type_id": int(tensor.tensor_type),
                                  "type": tensor.tensor_type.name, "shape": shape + [1],
                                  "expert_stride_bytes": int(tensor.n_bytes) // shape[2],
                                  "source": info, "first_last_128_sha256": samples}
        del reader
        if len(found) == len(wanted):
            break
    require(set(found) == wanted, "missing expert tensors: " + str(sorted(wanted - found.keys())))
    return {str(layer): [found[f"blk.{layer}.ffn_{role}_exps.weight"] for role in ("up", "gate", "down")] for layer in layers}


def runtime_identity(probe):
    paths = {probe.resolve()}
    for line in command(["ldd", str(probe)]).splitlines():
        match = re.search(r"=> (/\S+)", line)
        if match and any(part in Path(match[1]).name for part in ("llama", "ggml")):
            paths.add(Path(match[1]).resolve())
    require(len(paths) >= 5, "probe must link identifiable shared llama/ggml backends")
    return [{"path": str(path), "sha256": sha_file(path)} for path in sorted(paths)]


def host_identity():
    cpu = Path("/proc/cpuinfo").read_text().split("\n\n")[0]
    return {"kernel": platform.release(), "machine": platform.machine(), "cpuinfo_sha256": hashlib.sha256(
            "\n".join(line for line in cpu.splitlines() if line.startswith(("model name", "flags", "microcode", "cpu family", "model\t"))).encode()).hexdigest(),
            "cpu_model": next(line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")),
            "affinity": sorted(os.sched_getaffinity(0)), "numa_online": read_optional("/sys/devices/system/node/online"),
            "governor": read_optional("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")}


def child_env(probe, gpu):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_", "CUDA_"))}
    env.update(CUDA_VISIBLE_DEVICES=gpu["uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID", LD_LIBRARY_PATH=str(probe.parent),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env


def validate_measurement(data, expected_cases):
    require(data.get("complete") is True, "probe incomplete: " + str(data.get("error")))
    require([(x["tokens"], x["picks"]) for x in data["cases"]] == expected_cases, "probe case mismatch")
    for row in data["cases"]:
        for mode in MODES:
            stats = row["modes"][mode]
            values = stats["samples_us"]
            require(len(values) >= 3 and all(math.isfinite(x) and x > 0 for x in values), "invalid timing samples")
            require(stats["min_us"] == min(values) and stats["max_us"] == max(values)
                    and stats["median_us"] == sorted(values)[len(values) // 2], "invalid timing statistics")
            require(math.isfinite(stats["max_abs_vs_gpu"]), "invalid output comparison")
    require(data["storage"]["bytes_per_expert_host_bundle"] > 0, "invalid storage bundle")


def pci_normalize(value):
    return value.lower().lstrip("0")


def run_probe(args, gpu, tensors, quick=False):
    cases = CASES[:2] if quick else CASES
    cfg = {"host_mask": args.host_mask, "threads": args.threads, "repeats": 3 if quick else args.repeats,
           "warmup": 1 if quick else 2, "cases": [{"tokens": n, "picks": k} for n, k in cases], "tensors": tensors}
    with tempfile.TemporaryDirectory(prefix="moe-cost-probe-") as tmp:
        config, output, log = (Path(tmp) / name for name in ("config.json", "output.json", "probe.log"))
        config.write_text(json.dumps(cfg))
        with log.open("w") as stream:
            process = subprocess.run([str(args.probe), str(config), str(output)], env=child_env(args.probe, gpu),
                                     stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
        require(process.returncode == 0, f"probe exited {process.returncode}: " + log.read_text()[-6000:])
        data = json.loads(output.read_text())
        validate_measurement(data, cases)
        require(pci_normalize(data["device"]["pci_bus_id"]) == pci_normalize(gpu["pci.bus_id"]), "probe GPU identity mismatch")
        data["probe_log"] = log.read_text()
        return data


def load_cache(path):
    require(path.stat().st_size <= 32 * 1024 * 1024, "cost cache too large")
    data = json.loads(path.read_text())
    require(data.get("format") == FORMAT and data.get("version") == VERSION, "not a supported cost cache; choose a new path")
    checksum = data.pop("checksum")
    require(checksum == digest(data), "cost-cache checksum mismatch; preserve it and choose a new path")
    return data


def publish(path, data):
    value = copy.deepcopy(data)
    value["checksum"] = digest(value)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temp = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def calibrate(args):
    available = gpus()
    selected = []
    for key in args.gpus.split(","):
        matches = [gpu for gpu in available if key in (gpu["index"], gpu["uuid"])]
        require(len(matches) == 1 and matches[0] not in selected, "invalid/duplicate GPU: " + key)
        selected.append(matches[0])
    tensors = model_tensors(args.model, args.layers)
    # Dynamic idle PCIe generation is not an identity key.
    identity = {"runtime": runtime_identity(args.probe), "frontend_sha256": sha_file(Path(__file__)),
                "host": host_identity(), "gpus": selected, "layers": tensors,
                "protocol": {"revision": 1, "cases": CASES, "threads": args.threads, "host_mask": args.host_mask,
                             "repeats": args.repeats, "warmup": 2, "input_seed": 42, "expert_ids": "(r*47+t*picks+j)%experts",
                             "child_env": {k: v for k, v in child_env(args.probe, selected[0]).items()
                                           if k.startswith(("LLAMA_", "GGML_", "CUDA_", "OMP_", "OPENBLAS_", "LD_LIBRARY_PATH"))},
                             "recheck_relative_tolerance": 0.25, "recheck_absolute_floor_us": 20}}
    # Each child gets its own physical UUID; keep the common protocol independent of ordering.
    identity["protocol"]["child_env"]["CUDA_VISIBLE_DEVICES"] = "per measurement GPU UUID"
    def check_unchanged():
        require(runtime_identity(args.probe) == identity["runtime"]
                and sha_file(Path(__file__)) == identity["frontend_sha256"]
                and model_tensors(args.model, args.layers) == tensors,
                "runtime/model changed during calibration; old cache preserved")
        require(host_identity() == identity["host"] and gpus() == available,
                "CPU/GPU/PCIe identity changed during calibration; old cache preserved")

    fingerprint = digest(identity)
    old = load_cache(args.cache) if args.cache.exists() else None
    data = {"format": FORMAT, "version": VERSION, "identity": identity, "identity_sha256": fingerprint,
            "created_at_utc": now(), "measurements": [], "policy_applied": False}
    reason = "missing" if old is None else "identity_changed"
    if old and old["identity_sha256"] == fingerprint and not args.force:
        require(digest(old["identity"]) == fingerprint, "cached identity checksum mismatch")
        rechecks, drift = [], False
        expected = {(gpu["uuid"], layer) for gpu in selected for layer in tensors}
        require({(row["gpu_uuid"], row["layer"]) for row in old["measurements"]} == expected, "cached measurement coverage mismatch")
        for row in old["measurements"]:
            gpu = next(gpu for gpu in selected if gpu["uuid"] == row["gpu_uuid"])
            quick = run_probe(args, gpu, tensors[row["layer"]], quick=True)
            comparisons = []
            for case in quick["cases"]:
                saved = next(item for item in row["result"]["cases"] if (item["tokens"], item["picks"]) == (case["tokens"], case["picks"]))
                for mode in MODES:
                    previous, current = saved["modes"][mode]["median_us"], case["modes"][mode]["median_us"]
                    within = abs(current - previous) <= max(20, previous * 0.25)
                    drift |= not within
                    comparisons.append({"tokens": case["tokens"], "picks": case["picks"], "mode": mode,
                                        "old_us": previous, "new_us": current, "within_tolerance": within})
            rechecks.append({"gpu_uuid": row["gpu_uuid"], "layer": row["layer"], "result": quick, "comparisons": comparisons})
        if not drift:
            old["last_recheck_utc"] = now()
            old["last_recheck"] = rechecks
            check_unchanged()
            publish(args.cache, old)
            return {"status": "reused_after_recheck", "cache": str(args.cache), "identity_sha256": fingerprint}
        reason = "recheck_drift"
        data["preceding_recheck"] = rechecks
    if args.force:
        reason = "forced"
    for gpu in selected:
        for layer, desc in tensors.items():
            print(f"calibrating GPU {gpu['index']} layer {layer}", file=sys.stderr, flush=True)
            result = run_probe(args, gpu, desc)
            data["measurements"].append({"gpu_uuid": gpu["uuid"], "layer": layer, "result": result})
    check_unchanged()
    data["completed_at_utc"] = now()
    data["refresh_reason"] = reason
    publish(args.cache, data)
    return {"status": "calibrated", "reason": reason, "cache": str(args.cache), "identity_sha256": fingerprint}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="first GGUF shard; read only")
    parser.add_argument("--cache", required=True, type=Path, help="own versioned cost cache; atomic updates")
    parser.add_argument("--gpus", required=True, help="physical nvidia-smi indices or full UUIDs, comma separated")
    parser.add_argument("--layers", default="0,39", help="target layer IDs, comma separated")
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--host-mask", type=int, default=6, help="host projections: up=1, gate=2, down=4")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=600, help="timeout per GPU/layer probe, seconds")
    parser.add_argument("--probe", type=Path, default=Path(__file__).resolve().parents[1] / "build-aliceai/bin/llama-moe-cache-calibrate")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    lock = None
    locked = False
    try:
        args.layers = [int(x) for x in args.layers.split(",")]
        require(args.layers and len(set(args.layers)) == len(args.layers) and all(0 <= x < 40 for x in args.layers), "invalid target layers")
        require(1 <= args.threads <= len(os.sched_getaffinity(0)), "threads must fit CPU affinity")
        require(args.repeats >= 3 and args.repeats <= 99 and args.repeats % 2 == 1, "repeats must be odd, 3..99")
        require(1 <= args.host_mask <= 7 and args.timeout > 0, "invalid host-mask/timeout")
        args.probe = args.probe.resolve(strict=True)
        args.cache = args.cache.absolute()
        require(not args.cache.is_symlink(), "cost cache must not be a symlink")
        require(args.cache.parent.is_dir(), "cost cache parent does not exist")
        lock = Path(str(args.cache) + ".lock")
        lock.mkdir()
        locked = True
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        print(json.dumps(calibrate(args)))
    except (OSError, ValueError, KeyError, TypeError, ImportError, StopIteration, subprocess.SubprocessError) as error:
        parser.exit(1, f"calibration failed: {error}\n")
    finally:
        if locked:
            # Only remove a lock acquired by this process.
            lock.rmdir()


if __name__ == "__main__":
    main()
