# Upstream #31350 and #31368: rollback snapshot bounds, 2026-11-10

Status: adapted core source and targeted synthetic gates pass; committed at the user's request. Branch my_build, base7cf812027. Narrow GLM compatibility hooks stay with the unfinished working-tree implementation and are not staged with this core port. The preceding #31005 port and its report were committed at the user's request as7cf812027. No full-model benchmark, installed binary replacement, default promotion, push or merge is part of this stage.

## Problem and port

Upstream7df8b876b91208a86a6d792e1140b3deec14535b (#31350) restores the speculative checkpoint according to the complete draft length, rather than just the rejected tail. When a draft is longer than n_rs_seq, a short rejection does not guarantee a valid recurrent snapshot. Checkpoint creation already uses the full draft length; the acceptance path now uses that same criterion. The server translation unit and executable compile in the isolated build; an HTTP long-draft integration run was not performed.

Upstreamab3768cdc24644bdc285bc07b575ca367b1e1dd9 (#31368) refuses rollback beyond the snapshots guaranteed by the last ubatch touching each sequence:

| Memory | Last ubatch has n tokens of this sequence | Maximum rollback |
| --- | --- | --- |
| DSV4/DSV41 compressed state | Any n>0 | min(n,n_rs_seq), including the state before that ubatch |
| Recurrent/GDN state | n<=n_rs_seq+1 | n-1 |
| Recurrent/GDN state | n>n_rs_seq+1 | 0, only the final state is guaranteed |

Our existing single-use pending rollback and shared recurrent-cell refusal remain. DSV bounds reset after clear, destination copy and full/partial restore, including the local partial-restore failure cleanup. Recurrent bounds are cleared with discarded cells and move with a cell during reordering; the latter is a local adaptation beyond the upstream diff. Serialized state formats and CUDA arithmetic are unchanged.

## Compatibility with unfinished GLM work

The working tree already contains experimental GLM scalar windows and a two-stage token executor. They deliberately collect snapshots across single-token jobs. Applying the generic ubatch bound alone would reject their valid rollbacks, because each scalar apply advertises depth0.

A small set_rs_max helper publishes the manual window's depth only after its successful completion. The scalar completion path publishes total_rows-1; the two-stage executor publishes width-1 after draining and checking worker errors. Existing failed-window cleanup still clears memory. No scheduler ownership, kernel, flag eligibility, sampling or default is changed.

These compatibility edits are narrowly added to the already-dirty src/llama-context.cpp and untracked src/llama-glm-pipeline.cpp. The archive separates their exact delta from the pre-existing GLM work; this stage does not claim that whole experimental implementation as an upstream port. All19 other previously modified/untracked files retain their starting SHA256. Other GLM work must remain separate when staging the next commit.

## Gates

| Check | Result |
| --- | --- |
| Baseline negative control, CPU | DeepSeek41 and GLM5Next both fail the new oracle with rollback past last ubatch was accepted |
| Isolated compiler gate | llama-server, test-llama-archs and test-recurrent-state-rollback build successfully |
| Architecture fixtures, CPU | DeepSeek4, DeepSeek41, GLM5Next and Qwen4Exp pass bounds/state tests; Qwen3Next passes as an RS-disabled control |
| Architecture fixtures, Ada and Ampere | Four architectures pass on each device; valid replay, failed-restore recovery, source copy preservation and existing architecture gates pass |
| Explicit mixed layer split | DeepSeek41 and GLM5Next pass with first/last trunk devices CUDA0/CUDA1, not merely two visible devices |
| Existing recurrent-state executable, CPU | All four saved tiny GGUF models pass; no non-recurrent skip |
| GLM manual scalar window, CPU | 56 window/acceptance cases,10 controlled failures/retries,5 fallbacks; exact logits and state |
| GLM manual two-stage window, CPU | Same56/10/5 gates pass exactly |
| GLM manual two-stage window, Ada+Ampere | Same56/10/5 gates pass exactly |
| Formatting and source preservation | git diff --check passes; local core and GLM compatibility patches are recorded separately |

New architecture checks use n_rs_seq5 and widths1/2/6/7. Invalid rollback must leave positions and serialized state unchanged; the deepest valid rollback and its replay are checked, as are stacked refusal, dirty full/partial restore, truncated restore/recovery, destination copies, source preservation and another sequence's snapshots. Existing CPU-reference/roundtrip/MTP and feature checks run where the architecture fixture supports them. Manual GLM checks use RS1/2/4/7, changing widths2..8, every retained prefix, chained windows, fault injection and fallback controls. A passing synthetic fixture does not establish full trained-model quality.

The existing recurrent test had invalid expectations: rolling back several single-token decodes, and restoring the state before the first token of a recurrent ubatch. It now asserts refusal of stale rollback and rebuilds a window when a fresh snapshot is needed; multi-sequence replay retains an anchor before its rejected tail. No assertion is removed to conceal a runtime mismatch. Tests extend existing files only.

## Build and test isolation

The private source snapshot includes the existing working-tree changes, plus this port. Baseline differs by the five original runtime files, while retaining the regression oracle. Private CPU libllama/common/GGML modules were compiled with dynamic backend loading. The architecture executable needs explicit backend registration in that configuration; a startup-only shim loads the private CPU backend and registers the installed CUDA backend. It does not intercept graph evaluation. A separate, recorded compute-failure shim is used only for the controlled manual-window failure gates.

GPU UUIDs: GPU-9e5fb796-93b1-b72d-1fd8-438d93aac573 (Ada) and GPU-a84e3bf7-a50b-c880-8a79-7a7f81544783 (Ampere), CUDA_DEVICE_ORDER=PCI_BUS_ID. The obsolete permutation is not used. GPU absence/occupancy is checked before the owned GPU runs. These small tests do not certify a full target+draft512MiB memory floor or production context fit.

Initial harness/setup attempts are retained separately: an incompatible GGML_NATIVE/DL CMake combination, missing explicit backend registration, an RS-disabled Qwen3Next oracle, a server-only CLI flag, the canonical glm5-next filename, a missing C++ include, and a derived scalar fault assertion that incorrectly required the pipeline's exception handling. The final scalar fixture permits the existing propagated exception but still requires cleanup and exact recovery; the pipeline fixture still requires its error return. These are not claimed as runtime repairs. Earlier two-visible-device tests did not split the tiny trunk; the final explicitly weighted layer split confirms a real CUDA0-to-CUDA1 path.

## Evidence and limits

Local archive: /home/sergei/_my_old/_my_deepseek41/upstream-rollback-20261110. It holds upstream patches, before/after core sources, separate GLM compatibility patch, private baseline/candidate modules, build commands/logs, test helpers, failures and final results, model hashes and a SHA256 manifest. Large generated models and build intermediates remain under the recorded /tmp path and are reproducible from the existing architecture generator and seed1234. Only REPORT.md is intended for Git.

Full Q3 DSpark K3/K5 target/draft/tail-seeding and server long-draft accepted-token tests, text/code NLL/PPL and native PP/TG were not run in this stage. No speed gain, repair of the historical floating6 divergence, broader context fit or retained-plan resource coverage is inferred. Manual-window compatibility is covered by the recorded synthetic windows; the full GLM DFlash consumer and native warmup questions remain separate. Installed libraries and presets are unchanged.

Next: review this pair and its compatibility hooks, then the separately scoped #31584 strided-view bounds port. Keep the planned full-model checkpoint/DSpark acceptance gates visible before deployment; do not substitute the compile/operator screen for them.
