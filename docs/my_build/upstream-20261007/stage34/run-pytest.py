from pathlib import Path
import os,sys
R=Path(__file__).resolve().parent;T=Path('/home/sergei/Github/llama.cpp/tools/server/tests')
os.chdir(R);(R/'tmp').mkdir(exist_ok=True)
p=R/'tmp/stories15M-q4_0.gguf'
if not p.exists():p.symlink_to('/home/sergei/Github/llama.cpp/build-glm53/tinyllamas/stories15M-q4_0.gguf')
sys.path.insert(0,str(T));import utils
# This selected test uses only the existing local model; do not prefetch all unrelated server presets.
utils.ServerPreset.load_all=staticmethod(lambda:None)
os.environ.update(LLAMA_SERVER_BIN_PATH=str(R/'candidate-bin/llama-server'),LD_LIBRARY_PATH=str(R/'candidate-bin'),N_GPU_LAYERS='0',PORT='39867',CUDA_VISIBLE_DEVICES='')
import pytest
raise SystemExit(pytest.main([str(T/'unit/test_speculative.py')+'::test_slot_save_restore_draft','-q','-o','cache_dir='+str(R/'pytest-cache'),'--basetemp='+str(R/'pytest-tmp')]))
