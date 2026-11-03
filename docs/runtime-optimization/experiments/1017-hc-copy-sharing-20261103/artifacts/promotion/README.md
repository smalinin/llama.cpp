# Default ON promotion

User approved default ON after measured opt-in phase. Only selector delta is DEFAULT_ON.patch; full final source patch is PROMOTED.patch. Previous MEASURED.patch/after.cpp/results and their hashes remain unchanged.

Build with matching branch/internal headers and libraries. scheduler.cpp updates only expected unset behavior, then repeats OFF/ON/default negative/rotation/failure controls and fixed-graph capture. run.py reuses the parent metadata helper (same headers/API), dynamically loading promoted libraries; no_alloc/default/explicit1/OFF fixed-layout checks. Raw operator/correctness/native27request evidence remains in parent. Long tests were not repeated for selector promotion.

Unset or1 selects sharing subject to guards;0 disables. No commit/deploy. summary.json records final runtime/library and helper hashes.
