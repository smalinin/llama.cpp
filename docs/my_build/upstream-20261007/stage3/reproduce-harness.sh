#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/sergei/Github/llama.cpp
review_dir=/home/sergei/_my_sync/llama_upstream_review/stage3
before_dir=/home/sergei/_my_sync/llama_upstream_review/stage2/bin
fixtures="$repo_dir/build-glm53/tests/test-models"
common_flags=(-std=c++17 -O2 -I"$repo_dir/include" -I"$repo_dir/src" -I"$repo_dir/ggml/include" -I"$repo_dir/common")
common_libs=(-lllama -lggml -lggml-base -lggml-cpu)
for target in state-fault state-api state-compat file-truncated; do
    c++ "${common_flags[@]}" "$review_dir/$target.cpp" -L"$review_dir/bin" -Wl,-rpath,"$review_dir/bin" "${common_libs[@]}" -o "$review_dir/$target-reproduced"
done
c++ "${common_flags[@]}" "$review_dir/rotation-only.cpp" -L"$review_dir/bin" -Wl,-rpath,"$review_dir/bin" -lllama-common "${common_libs[@]}" -o "$review_dir/rotation-only-reproduced"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,3 LD_LIBRARY_PATH="$review_dir/bin" "$review_dir/state-fault-reproduced" "$fixtures"
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$review_dir/bin" "$review_dir/state-api-reproduced" "$fixtures"
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$review_dir/bin" "$review_dir/rotation-only-reproduced" "$fixtures/llama-dense.gguf"
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$review_dir/bin" "$review_dir/file-truncated-reproduced" "$fixtures/llama-dense.gguf" "$review_dir/reproduced-truncated.bin"
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$review_dir/bin" "$review_dir/state-compat-reproduced" "$fixtures/llama-dense.gguf" "$review_dir/current-reproduced" save
CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$review_dir/bin" "$review_dir/state-compat-reproduced" "$fixtures/llama-dense.gguf" "$review_dir/current-reproduced" match
# Expected failures on the old library, run separately:
# CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$before_dir" "$review_dir/state-fault-reproduced" "$fixtures" llama-dense
# CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$before_dir" "$review_dir/rotation-only-reproduced" "$fixtures/llama-dense.gguf"
# ulimit -c 0; CUDA_VISIBLE_DEVICES='' LD_LIBRARY_PATH="$before_dir" "$review_dir/file-truncated-reproduced" "$fixtures/llama-dense.gguf" "$review_dir/reproduced-truncated-old.bin"
# Real-model runner creates its output directory and must run without another GPU test:
# python3 "$review_dir/run-real-state.py"
# python3 "$review_dir/run-real-state.py" --qwen-q8
