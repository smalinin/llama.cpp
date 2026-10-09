# Stage 34 review artifacts

Base: my_build at 45cbfc78d. The user reviewed and approved these changes for commit. Installed binaries and active presets are unchanged.

- `build.py`: compile the changed MMF and server objects, expand CMake link response files and relink isolated libraries over Stage33 final binaries. `build-manifest.json` records exact commands, source and library hashes. Requires the preserved build-glm53 object tree.
- `cuda-strides.cpp`: Stage4 CPU/CUDA comparison harness extended with 260-byte row stride and adjacent layouts. Allocate and initialize the physical backing tensors before creating views. Run `cuda-strides strides output.bin` with `LD_LIBRARY_PATH=candidate-bin` and a GPU UUID in CUDA_VISIBLE_DEVICES. Final logs: strides-ada.log and strides-rtx3090.log, 288 cases each.
- `build-test.py`: build the existing test-backend-ops with the new four cases. Select with `test -b CUDA0 -o MUL_MAT -p k_v=130`.
- `run-slot.py`: real-model save/restore, exact continuation, cache hits, optional restart/fault checks. `run-final-slots.py` lists final invocations. GLM-DSA uses the user's IQ2 model, DeepSeek uses the Stage33 long token prompt.
- `run-pytest.py`: execute the added existing server test against the local TinyLlama file, using a temporary Python environment. Skip unrelated preset prefetch, without changing the selected test. `native-compat.py` checks reading a speculative file on a native server with matching FA settings.
- `summarize.py`: verify final results and source hashes. `summary.json` is the verified result set; preliminary runs are excluded.

Final real-model directories: qwen-final-v2, glm5next-final, glm-dsa-final, deepseek-final, native-final. qwen-before is the pre-fix reference. Earlier qwen-final was checked before the final restore clearing / atomic save refinement and is not counted in the final matrix.

No model weights, compiled binaries, tensor dumps or slot blobs are included in the repository archive. They remain in the workspace stage34 directory. The installed server and active model presets are unchanged.
