#pragma once
#include <cstdint>
static const char * counter_names[]={"graphs","nodes","float_matmul","float_single_row_matmul","moe_matmul","get_rows","backend_sync","scheduler_sync","event_sync","event_wait","copy_async_api","copy_sync_api","cuda_graph_launch","cuda_graph_instantiate","cuda_kernel_launch","prepare_inputs","allocate_graph","quant_matmul"};
constexpr int counter_count=sizeof(counter_names)/sizeof(counter_names[0]);
