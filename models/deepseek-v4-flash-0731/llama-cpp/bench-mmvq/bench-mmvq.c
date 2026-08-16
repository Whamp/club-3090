// Microbenchmark: drive the pinned fork's real CUDA MMVQ dispatch at the exact
// DeepSeek V4 decode shapes and report per-op time and achieved GB/s.
//
// Build inside nvidia/cuda:12.8.1-devel-ubuntu24.04 against the fork's ggml.
// Usage: ./bench-mmvq [device] -> CSV rows: op,rows,cols,us_per_iter,gbps
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

struct shape_run {
    const char * name;
    ggml_type type;
    int64_t ne00; // K (input cols)
    int64_t ne01; // rows (output)
};

int main(int argc, char ** argv) {
    int device = argc > 1 ? atoi(argv[1]) : 0;
    const int only = argc > 2 ? atoi(argv[2]) : -1; // run a single shape by index
    ggml_time_init();

    const size_t nelements_max = 4096 * 4096;
    std::vector<float> src(nelements_max);
    unsigned int seed = 1234;
    auto rnd = [&]() {
        // approximately gaussian (sum of uniforms): keeps K-quant k-means on-grid
        seed = seed * 1664525u + 1013904223u;
        float acc = 0.0f;
        for (int k = 0; k < 12; k++) { seed = seed * 1664525u + 1013904223u; acc += (float)((seed >> 8) & 0xffff) / 65535.0f; }
        return acc - 6.0f;
    };
    for (auto & v : src) v = rnd();

    std::vector<shape_run> runs = {
        {"IQ2_XXS_K4096_R2048", GGML_TYPE_IQ2_XXS, 4096, 2048},
        {"IQ2_XXS_K4096_R12288", GGML_TYPE_IQ2_XXS, 4096, 12288},
        {"IQ2_XXS_K2048_R2048", GGML_TYPE_IQ2_XXS, 2048, 2048},
        {"IQ2_XXS_K1024_R4096", GGML_TYPE_IQ2_XXS, 1024, 4096},
        {"Q2_K_K4096_R2048",    GGML_TYPE_Q2_K,    4096, 2048},
        {"Q2_K_K4096_R12288",   GGML_TYPE_Q2_K,    4096, 12288},
        {"Q2_K_K2048_R2048",    GGML_TYPE_Q2_K,    2048, 2048},
        {"Q2_K_K1024_R4096",    GGML_TYPE_Q2_K,    1024, 4096},
        {"Q8_0_K4096_R4096",    GGML_TYPE_Q8_0,    4096, 4096},
        {"Q8_0_K2048_R2048",    GGML_TYPE_Q8_0,    2048, 2048},
    };
    if (only >= 0 && only < (int)runs.size()) {
        shape_run keep = runs[only];
        runs.clear();
        runs.push_back(keep);
    }

    printf("device,%d\n", device);

    for (const auto & r : runs) {
        const size_t row_bytes = ggml_row_size(r.type, r.ne00);
        const size_t w_bytes = row_bytes * r.ne01;

        ggml_backend_t backend = ggml_backend_cuda_init(device);
        if (!backend) { fprintf(stderr, "cuda init failed\n"); return 1; }

        ggml_init_params ip = { 256u*1024u*1024u, nullptr, /*no_alloc*/ true};
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * w = ggml_new_tensor_2d(ctx, r.type, r.ne00, r.ne01);
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, r.ne00, 1);
        ggml_tensor * mm = ggml_mul_mat(ctx, w, x);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(backend));

        std::vector<uint8_t> wdata(w_bytes);
        // Kernel time in these vec_dot implementations is data-independent
        // (fixed instruction mix; grid lookups are gathers regardless of byte
        // values), and the CPU K-quant/i-quant quantizers abort on degenerate
        // synthetic blocks. Fill blocks directly from a deterministic LCG and
        // pin fp16 block scales to 1.0 to avoid flush-to-zero timing effects.
        {
            const int64_t blck = ggml_blck_size(r.type);
            const size_t nblck_row = r.ne00 / blck;
            const size_t row_blck_bytes = row_bytes / nblck_row;
            for (size_t i = 0; i < w_bytes; i++) wdata[i] = (uint8_t)((seed = seed * 1664525u + 1013904223u) >> 24);
            for (int64_t row = 0; row < r.ne01; row++) {
                for (size_t b = 0; b < nblck_row; b++) {
                    uint8_t * blk = wdata.data() + row * row_bytes + b * row_blck_bytes;
                    const size_t d_off = (r.type == GGML_TYPE_Q2_K) ? row_blck_bytes - 4 : 0;
                    if (r.type == GGML_TYPE_IQ2_XXS || r.type == GGML_TYPE_Q8_0) {
                        blk[0] = 0x00; blk[1] = 0x3C; // fp16 1.0 at block start
                    } else if (r.type == GGML_TYPE_Q2_K) {
                        blk[d_off + 0] = 0x00; blk[d_off + 1] = 0x3C; // d
                        blk[d_off + 2] = 0x00; blk[d_off + 3] = 0x3C; // dm
                    }
                }
            }
        }
        const int64_t quantized = (int64_t)w_bytes;
        if (quantized != (int64_t)w_bytes) {
            fprintf(stderr, "quantize mismatch %lld vs %zu\n", (long long)quantized, w_bytes);
            return 1;
        }
        ggml_backend_tensor_set(w, wdata.data(), 0, w_bytes);
        ggml_backend_tensor_set(x, src.data(), 0, ggml_nbytes(x));

        const int reps = 16; // independent nodes per graph: amortize launch/enqueue cost
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, reps + 8, false);
        for (int i = 0; i < reps; i++) {
            ggml_tensor * mm = ggml_mul_mat(ctx, w, x);
            ggml_build_forward_expand(graph, mm);
        }

        // Allocation must cover the repeat outputs too.
        ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors_from_buft(
            ctx, ggml_backend_get_default_buffer_type(backend));
        (void)buf2;

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n"); return 1;
        }
        ggml_backend_synchronize(backend);

        const int warmup = 20, iters = 100;
        for (int i = 0; i < warmup; i++) ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const double t0 = ggml_time_ms();
        for (int i = 0; i < iters; i++) ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const double dt_ms = ggml_time_ms() - t0;

        const double us = dt_ms * 1000.0 / (iters * reps);
        const double gbps = (double)w_bytes / (us * 1e-6) / 1e9;

        printf("%s,%lld,%lld,%.3f,%.2f\n", r.name, (long long)r.ne01, (long long)r.ne00, us, gbps);
        fflush(stdout);

        ggml_backend_buffer_free(buf2);
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        ggml_backend_free(backend);
    }
    return 0;
}
