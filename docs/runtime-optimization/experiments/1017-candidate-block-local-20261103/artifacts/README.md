# Reproduction and artifact scope

Base and exact five-source delta: config.json / FINAL.patch / provenance.json. Built helper binaries and uncompressed large probes remain at /tmp/dsv41-candidate-block-local-20261103; helper sources, arguments and results are archived here. Recompile against matching current internal headers and libraries; do not run historical helpers blindly against another branch. source/library hashes identify the measured frozen build.

Serial phases: build -> external operator -> fresh OFF/ON precision -> idle no_alloc/negative controls -> clean native OFF/ON/OFF -> guarded actual176128/read-only copy proof -> analyze-audit.py/finalize.py. memory_monitor.py maps CUDA/NVML by UUID/PCI, sampled floor512MiB. Full argv/env/model paths/raw IDs and corpus inputs are in config/precision/bench manifests. Tests must not overlap another GPU workload. No commit, model mutation, default promotion or deployment is implied.

Large JSONL/log files are losslessly gzipped; JSON results retain their exact contents. metadata-initial retains first free-before samples that overlapped the operator; only fresh idle metadata informed admission. pool0 controls are metadata-only. Copy proof is logical API requests, not physical DMA. See REPORT for limitations. JSON paths inside raw manifests refer to the original /tmp run and have not been rewritten.
