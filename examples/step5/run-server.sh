#!/usr/bin/env bash
set -euo pipefail

step5_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
step5_model="${STEP5_MODEL:-$step5_repo/models/Step-5-Preview-GGUF/Step5-Preview-Q3_K_M.gguf-00001-of-00008.gguf}"
step5_binary="${STEP5_SERVER:-$step5_repo/build/bin/llama-server}"
step5_attention="${STEP5_ATTENTION:-}"
if [[ -z "$step5_attention" ]]; then
  if [[ "${LLAMA_STEP5_EXPERIMENTAL_CSA:-0}" == 1 ]]; then step5_attention=csa
  elif [[ "${LLAMA_STEP5_LEGACY_ATTENTION:-0}" == 1 ]]; then step5_attention=legacy
  else step5_attention=dense
  fi
fi
step5_context="${STEP5_CONTEXT:-144000}"
step5_fit="${STEP5_FIT:-on}"
if [[ -z "${STEP5_FIT:-}" && ( -n "${STEP5_GPU_LAYERS:-}" || -n "${STEP5_TENSOR_SPLIT:-}" ) ]]; then step5_fit=off; fi
step5_mtp="${STEP5_MTP:-3}"
step5_cache_type="${STEP5_CACHE_TYPE:-f16}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export LLAMA_MTP_ADAPTIVE="${LLAMA_MTP_ADAPTIVE:-1}"
unset LLAMA_STEP5_EXPERIMENTAL_CSA LLAMA_STEP5_EXPERIMENTAL_DENSE LLAMA_STEP5_EXPERIMENTAL_SPARSE LLAMA_STEP5_LEGACY_ATTENTION
case "$step5_attention" in
  csa) export LLAMA_STEP5_EXPERIMENTAL_CSA=1 ;;
  dense) export LLAMA_STEP5_EXPERIMENTAL_DENSE=1 ;;
  legacy) export LLAMA_STEP5_EXPERIMENTAL_DENSE=1 LLAMA_STEP5_LEGACY_ATTENTION=1 ;;
  *) echo 'STEP5_ATTENTION must be csa, dense or legacy' >&2; exit 2 ;;
esac
case "$step5_fit" in on|off) ;; *) echo 'STEP5_FIT must be on or off' >&2; exit 2 ;; esac
case "$step5_mtp" in 0|1|2|3) ;; *) echo 'STEP5_MTP must be 0, 1, 2 or 3' >&2; exit 2 ;; esac
case "$step5_cache_type" in f16|q8_0) ;; *) echo 'STEP5_CACHE_TYPE must be f16 or q8_0' >&2; exit 2 ;; esac
if [[ ! -x "$step5_binary" ]]; then echo "Server binary is not executable: $step5_binary" >&2; exit 2; fi
if [[ ! -f "$step5_model" ]]; then echo "Model not found: $step5_model" >&2; exit 2; fi

step5_placement=(--fit "$step5_fit" --fit-target "${STEP5_FIT_TARGET:-2048}" --fit-ctx "$step5_context")
if [[ "$step5_fit" == off ]]; then step5_placement+=(-ngl "${STEP5_GPU_LAYERS:-72}")
elif [[ -n "${STEP5_GPU_LAYERS:-}" ]]; then step5_placement+=(-ngl "$STEP5_GPU_LAYERS")
fi
if [[ -n "${STEP5_TENSOR_SPLIT:-}" ]]; then step5_placement+=(-ts "$STEP5_TENSOR_SPLIT"); fi
if [[ -n "${STEP5_OVERRIDE_TENSORS:-}" ]]; then step5_placement+=(-ot "$STEP5_OVERRIDE_TENSORS"); fi
step5_spec_type=none
if ((step5_mtp > 0)); then step5_spec_type=draft-mtp; fi
step5_spec_args=()
step5_has_spec_type=0
step5_has_spec_max=0
for step5_arg in "$@"; do
  case "$step5_arg" in
    --spec-type|--spec-type=*) step5_has_spec_type=1 ;;
    --spec-draft-n-max|--spec-draft-n-max=*) step5_has_spec_max=1 ;;
  esac
done
if ((step5_has_spec_type == 0)); then step5_spec_args+=(--spec-type "$step5_spec_type"); fi
if ((step5_has_spec_max == 0 && step5_has_spec_type == 0)); then step5_spec_args+=(--spec-draft-n-max "$step5_mtp"); fi
step5_spec_label="$step5_mtp"
if ((step5_has_spec_type || step5_has_spec_max)); then step5_spec_label='CLI override'; fi
step5_vision=()
if [[ -n "${STEP5_MMPROJ:-}" ]]; then
  if [[ ! -f "$STEP5_MMPROJ" ]]; then echo "Projector not found: $STEP5_MMPROJ" >&2; exit 2; fi
  step5_vision=(--mmproj "$STEP5_MMPROJ" --no-mmproj-offload)
fi
printf 'Step-5 attention=%s, context=%s, MTP=%s, fit=%s, KV=%s\n' "$step5_attention" "$step5_context" "$step5_spec_label" "$step5_fit" "$step5_cache_type" >&2
exec "$step5_binary" -m "$step5_model" "${step5_vision[@]}" \
  -c "$step5_context" -b "${STEP5_BATCH:-2048}" -ub "${STEP5_UBATCH:-512}" \
  "${step5_placement[@]}" --load-mode "${STEP5_LOAD_MODE:-none}" \
  -t "${STEP5_THREADS:-20}" -tb "${STEP5_THREADS_BATCH:-20}" \
  --no-op-offload -fa on --cache-type-k "$step5_cache_type" --cache-type-v "$step5_cache_type" \
  --no-context-shift --no-warmup --parallel 1 \
  --host "${STEP5_HOST:-127.0.0.1}" --port "${STEP5_PORT:-8085}" --alias step5 \
  --temp 0.7 --top-p 0.95 --jinja --chat-template-kwargs '{"reasoning_effort":"medium"}' \
  "${step5_spec_args[@]}" "$@"
