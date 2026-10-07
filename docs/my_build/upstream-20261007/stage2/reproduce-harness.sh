#!/usr/bin/env bash
set -euo pipefail
repo_dir=/home/sergei/Github/llama.cpp
review_dir=/home/sergei/_my_sync/llama_upstream_review/stage2
before_dir=/home/sergei/_my_sync/llama_upstream_review/stage1/bin
header_dir=$(mktemp -d /tmp/llama-stage2-headers.XXXXXX)
trap 'rm -rf "$header_dir"' EXIT
git -C "$repo_dir" show 00091b7f2694f03839aec08248b11ed236619641:src/llama-graph.h > "$header_dir/llama-graph.h"
common_flags=(-std=c++17 -O2 -I"$repo_dir/include" -I"$repo_dir/src" -I"$repo_dir/ggml/include")
common_libs=(-lllama -lggml -lggml-base -lggml-cpu)
c++ "${common_flags[@]}" "$review_dir/hidden-order.cpp" -L"$review_dir/bin" -Wl,-rpath,"$review_dir/bin" "${common_libs[@]}" -o "$review_dir/hidden-order-reproduced"
c++ "${common_flags[@]}" "$review_dir/recurrent-reserve.cpp" -L"$review_dir/bin" -Wl,-rpath,"$review_dir/bin" "${common_libs[@]}" -o "$review_dir/recurrent-reserve-reproduced"
c++ -DBEFORE_STAGE2 -I"$header_dir" "${common_flags[@]}" "$review_dir/recurrent-reserve.cpp" -L"$before_dir" -Wl,-rpath,"$before_dir" "${common_libs[@]}" -o "$review_dir/recurrent-reserve-before-reproduced"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,3 "$review_dir/hidden-order-reproduced" "$repo_dir/build-glm53/tests/test-models"
GGML_SCHED_DEBUG_REALLOC=2 "$review_dir/recurrent-reserve-reproduced"
# Expected baseline failure, run separately:
# GGML_SCHED_DEBUG_REALLOC=1 "$review_dir/recurrent-reserve-before-reproduced"
