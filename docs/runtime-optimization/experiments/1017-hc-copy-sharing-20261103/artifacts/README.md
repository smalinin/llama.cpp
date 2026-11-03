# Evidence phases

Parent summary.json/config.json/MEASURED.patch/ggml-backend.after.cpp and native/operator/bench results describe the measured opt-in build, whose unset selector was OFF. They are immutable historical evidence.

The current default is ON after the user requested promotion. promotion/ contains the separate selector delta, full promoted patch, source/library hashes and new default-vs-explicit1 checks. Use promotion/summary.json for current promotion status, not the historical parent summary default. No long numeric or speed series was rerun for that selector-only change.

For reproduction use a scratch directory and matching source/libraries/internal headers; remap recorded absolute paths in helper sources/runners. Parent config records frozen argv/environment and measured hashes. Apply MEASURED.patch to base9b74cc6e6 for the exact measured source; use promotion/PROMOTED.patch for final defaultON source. Explicit1 selects the same algorithm in both builds. Do not execute archive/finalize/update scripts against immutable evidence paths. Model payloads and compiled helpers are external; see provenance locators/hashes.
