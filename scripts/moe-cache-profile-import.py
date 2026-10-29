#!/usr/bin/env python3
"""Import the target part of a Maya DeepSeek41 GGUF route profile."""
import argparse
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile

MAX_COUNT = (1 << 64) - 1
PHASES = ("single_token", "short_batch", "prefill")
ROLES = {"up": "w3", "gate": "w1", "down": "w2"}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fnv(data, value=14695981039346656037):
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & MAX_COUNT
    return value


def read_template(path):
    require(not Path(str(path) + ".lock").exists(), "template has an active/stale writer lock")
    raw = path.read_bytes()
    require(0 < len(raw) <= 16 * 1024 * 1024, "invalid template size")
    data = json.loads(raw)
    checksum = data.pop("checksum")
    require(checksum == f"{fnv(canonical(data)):016x}", "template checksum mismatch")
    require(data["format"] == "llama-moe-route-profile" and type(data["version"]) is int
            and data["version"] in (1, 2) and data["namespace"] == "target", "unsupported template format")
    require(data["identity"]["architecture"] == "deepseek41", "only DeepSeek41 target imports are supported")
    layout = data["identity"]["layout"]
    require(len(layout) == 40 and len(data["layers"]) == 40, "expected 40 target layers")
    for layer, (desc, row) in enumerate(zip(layout, data["layers"])):
        require(desc["layer"] == row["layer"] == layer, "invalid layer IDs")
        require(len(desc["tensors"]) == 3, "expected up/gate/down projections")
        for role, tensor in zip(ROLES, desc["tensors"]):
            require(tensor["name"] == f"blk.{layer}.ffn_{role}_exps.weight", "unexpected projection mapping")
            shape = [2304, 5120, 384, 1] if role == "down" else [5120, 2304, 384, 1]
            require(tensor["shape"] == shape, "unsupported target geometry")
        for phase in PHASES:
            values = row["counts"][phase]
            require(len(values) == 384 and all(type(v) is int and 0 <= v <= MAX_COUNT for v in values),
                    "invalid template route counts")
            require(type(row["calls"][phase]) is int and 0 <= row["calls"][phase] <= MAX_COUNT,
                    "invalid template observation count")
    return data, hashlib.sha256(raw).hexdigest()


def verify_pack(pack, identity):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
    from gguf import GGUFReader

    config_raw = (pack / "ds4.txt").read_bytes()
    config = dict(line.split(maxsplit=1) for line in config_raw.decode().splitlines() if len(line.split()) >= 2)
    for key, value in {"num_hidden_layers": "40", "n_routed_experts": "384", "hidden_size": "5120",
                       "moe_intermediate_size": "2304", "format": "gguf"}.items():
        require(config.get(key) == value, "Maya geometry/format mismatch: " + key)
    table_raw = (pack / "tensors.txt").read_bytes()
    files, pieces = {}, {}
    pattern = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.weight")
    for line in table_raw.decode().splitlines():
        if line.startswith("file "):
            _, index, name = line.split(maxsplit=2)
            require(int(index) not in files, "duplicate file index")
            files[int(index)] = (pack.parent / name).resolve()
        elif line.startswith("t "):
            fields = line.split()
            match = pattern.fullmatch(fields[1])
            if not match or int(match[1]) >= 40:
                continue
            require(len(fields) == 8, "unsupported expert descriptor")
            key = (int(match[1]), match[3], int(match[2]))
            require(key not in pieces, "duplicate expert ID")
            pieces[key] = (fields[2].lower(), *map(int, fields[3:]))
    require(len(pieces) == 40 * 384 * 3, "incomplete/extra target expert mapping")
    readers, tensors = {}, {}
    sample_hash = 14695981039346656037
    descriptors = 0
    for layer in identity["layout"]:
        for role, desc in zip(ROLES, layer["tensors"]):
            part = ROLES[role]
            first = pieces[(layer["layer"], part, 0)]
            dtype, file_id, offset, size, rows, columns = first
            require(file_id in files, "missing source file")
            path = files[file_id]
            if path not in readers:
                readers[path] = GGUFReader(path)
                tensors[path] = {t.name: t for t in readers[path].tensors}
            require(desc["name"] in tensors[path], "source GGUF lacks " + desc["name"])
            tensor = tensors[path][desc["name"]]
            shape = list(tensor.shape) + [1] * (4 - len(tensor.shape))
            require(shape == desc["shape"] and tensor.tensor_type.name.lower() == desc["type"].lower() == dtype,
                    "GGUF tensor shape/type mismatch")
            require(offset == tensor.data_offset and size * 384 == tensor.n_bytes,
                    "GGUF tensor offset/size mismatch")
            require([columns, rows] == shape[:2], "Maya gate/up/down axis mismatch")
            for expert in range(384):
                require(pieces[(layer["layer"], part, expert)] ==
                        (dtype, file_id, offset + expert * size, size, rows, columns), "expert ID/stride mismatch")
                descriptors += 1
            with path.open("rb") as weight:
                for position in (offset, offset + tensor.n_bytes - 128):
                    weight.seek(position)
                    sample = weight.read(128)
                    require(len(sample) == 128, "truncated source tensor")
                    sample_hash = fnv(sample, sample_hash)
    require(f"{sample_hash:016x}" == identity["weight_sample_fnv64"], "source weight samples do not match template")
    return {"ds4_sha256": hashlib.sha256(config_raw).hexdigest(),
            "tensors_sha256": hashlib.sha256(table_raw).hexdigest(),
            "checked_expert_slices": descriptors, "checked_weight_files": len(readers),
            "source_index_topk": int(config["index_topk"]), "runtime_topk_override_unknown": True,
            "identity_check": "routed GGUF names/types/shapes/offsets/all expert IDs and first/last128-byte samples"}


def make_profile(template, pack):
    data, template_hash = read_template(template)
    source = pack / "expert-profile.bin"
    raw = source.read_bytes()
    require(len(raw) in (40 * 384 * 4, 43 * 384 * 4), "expected 40x384 or 43x384 little-endian uint32 profile")
    values = struct.unpack("<" + "I" * (len(raw) // 4), raw)
    require(not any(values[40 * 384:]), "nonzero draft tail is outside this target-only importer")
    checks = verify_pack(pack, data["identity"])
    result = copy.deepcopy(data)
    result["version"] = 2
    for row in result["layers"]:
        start = row["layer"] * 384
        row["seed_counts"] = list(values[start:start + 384])
    result["seed_provenance"] = {
        "kind": "maya", "source_sha256": hashlib.sha256(raw).hexdigest(),
        "template_sha256": template_hash, "profile_rows": len(values) // 384,
        "target_layers": 40, "experts": 384, "source_profile_path": str(source.resolve()),
        "pack_path": str(pack.resolve()), "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "semantics": "Maya aggregate counts; phase, accepted/rejected split and bundled/learned history unknown",
        **checks,
    }
    result["checksum"] = f"{fnv(canonical(result)):016x}"
    return result


def publish(output, data):
    lock = Path(str(output) + ".lock")
    lock.mkdir()
    temp = None
    try:
        require(not output.exists(), "output exists; choose a new filename")
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=lock, delete=False) as stream:
            temp = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link publishes a complete file without replacing an existing path.
        os.link(temp, output)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True, help="closed runtime-generated target profile; preserved")
    parser.add_argument("--maya-pack", type=Path, required=True, help="maya-ds4 or maya-ds4-dspark directory; read only")
    parser.add_argument("--output", type=Path, required=True, help="new target profile filename; never overwritten")
    args = parser.parse_args()
    try:
        require(not args.output.resolve().is_relative_to(args.maya_pack.resolve()), "output must be outside the Maya pack")
        require(not args.output.exists() and args.output != args.template, "output must be a new file")
        result = make_profile(args.template, args.maya_pack)
        publish(args.output, result)
        print(json.dumps({"output": str(args.output), "seed_sha256": result["seed_provenance"]["source_sha256"],
                          "target_layers": 40, "experts": 384, "checks": result["seed_provenance"]["identity_check"]}))
    except (OSError, ValueError, KeyError, ImportError, TypeError) as error:
        parser.exit(1, f"import failed: {error}\n")


if __name__ == "__main__":
    main()
