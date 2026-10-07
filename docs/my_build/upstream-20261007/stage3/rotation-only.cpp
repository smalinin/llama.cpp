#define main save_load_suite_main
#include "/home/sergei/Github/llama.cpp/tests/test-save-load-state.cpp"
#undef main
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    common_init();
    ggml_backend_load_all();
    common_params params;
    params.model.path = argv[1];
    params.n_gpu_layers = 0;
    params.n_ctx = 64;
    auto init = common_init_from_params(params, true);
    if (!init->model()) return 2;
    return test_state_rotation(init->model(), params) ? 0 : 1;
}
