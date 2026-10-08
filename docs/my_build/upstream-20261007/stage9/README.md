# Stage 9 local diagnostics

Tested runtime: immutable `../stage8/candidate-bin`; installed server is unchanged.

- `current-server/`: three native and three DSpark N=3 diagnostic HTTP completions.
- `forced-current-native.i32`: current native output, used as identical teacher-forced input in all controls.
- `controls-output/`: widths 1/2/4, common scalar prefix of 16 tokens, full-layer feature export, float precision, F32 KV and non-Flash attention controls.
- `no-fusion-output/`: separate process with `GGML_CUDA_DISABLE_FUSION=1`.
- `n1-server/`: native vs DSpark N=1, four prompts and three measured repeats.
- `attention-capture-output/`: compressed-source layer 2 tensor inputs at the first decode call, followed by precision controls applied only after prefill.

Full logits, activations, binaries and server logs stay outside Git. Review evidence is copied to repository docs after all jobs finish. A matching greedy prefix alone does not establish logit equivalence or general correctness. Tensor capture is accepted as evidence about the uninstrumented graph only when all saved logits have the same SHA256 as the corresponding control.
