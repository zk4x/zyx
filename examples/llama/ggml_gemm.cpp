// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

// llama-3.2-3B GEMMs through ggml's CUDA backend, per projection and per
// phase.
//
// Mirrors `zyx-bench llama prefill_gemm` and `zyx-bench llama decode_gemm`: the
// same seven projections at the same lengths (`weight [n, k] x input [m, k]`,
// both f16), plus `lm_head` at decode length. This is the per-GEMM path
// llama.cpp itself runs (its f16 prefill goes `mul_mat` -> cuBLAS at these
// sizes), which is what the aggregate "16 TFLOP/s for llama.cpp" number
// decomposes into.
//
// Build against a llama.cpp checkout (same layout and command as `llama_ids`):
//
//   LLAMA_CPP=~/Dev/cpp/llama_cpp/llama.cpp
//   g++ -O2 examples/llama/ggml_gemm.cpp -o /tmp/opencode/ggml_gemm \
//     -I$LLAMA_CPP/include -I$LLAMA_CPP/ggml/include \
//     -L$LLAMA_CPP/build/bin -lllama -lggml -lggml-base -lggml-cuda \
//     -Wl,-rpath,$LLAMA_CPP/build/bin
//
// then `ggml_gemm`. Cycles = 200 iterations, best of 5 batches.

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdio>
#include <ctime>
#include <random>
#include <vector>

namespace {

constexpr int HIDDEN = 3072;
constexpr int INTERMEDIATE = 8192;
constexpr int HEADS = 24;
constexpr int KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int VOCAB = 128256;
constexpr int SEQ = 128;

constexpr int TRIALS = 5;
constexpr int REPS = 50;
constexpr int WARMUP = 20;

struct Proj {
    const char * name;
    int k;
    int n;
};

// `(name, k, n)` in the order a decoder block runs them; matches
// `zyx-bench/src/bench/llama/matmul.rs`.
constexpr Proj PROJS[] = {
    {"q_proj", HIDDEN, HEADS * HEAD_DIM},
    {"k_proj", HIDDEN, KV_HEADS * HEAD_DIM},
    {"v_proj", HIDDEN, KV_HEADS * HEAD_DIM},
    {"o_proj", HEADS * HEAD_DIM, HIDDEN},
    {"gate_proj", HIDDEN, INTERMEDIATE},
    {"up_proj", HIDDEN, INTERMEDIATE},
    {"down_proj", INTERMEDIATE, HIDDEN},
};

void fill_f16(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<ggml_fp16_t> buf(ggml_nelements(t));
    for (auto & v : buf) v = ggml_fp32_to_fp16(dist(rng));
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(buf[0]));
}

double now_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) + double(ts.tv_nsec) * 1e-9;
}

double measure(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < WARMUP; ++i) ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    double best = 1e30;
    for (int trial = 0; trial < TRIALS; ++trial) {
        const double start = now_seconds();
        for (int i = 0; i < REPS; ++i) ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        const double per = (now_seconds() - start) / REPS;
        if (per < best) best = per;
    }
    return best;
}

// `(phase, sequence length)`. Decode adds `lm_head`, exactly like
// `zyx-bench llama decode_gemm` (`matmul.rs::lm_head`).
struct Phase {
    const char * name;
    int seq;
    bool lm_head;
};
constexpr Phase PHASES[] = {{"prefill", SEQ, false}, {"decode", 1, true}};

} // namespace

int main() {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "ggml: no CUDA backend\n");
        return 1;
    }
    char desc[128] = {};
    ggml_backend_cuda_get_device_description(0, desc, sizeof(desc));
    std::printf("ggml %s, %s\n", ggml_backend_name(backend), desc);

    std::mt19937 rng(0);
    for (const Phase & phase : PHASES) {
        double total_s = 0.0;
        double total_flops = 0.0;
        double total_bytes = 0.0;
        for (int i = 0; i < (int) (sizeof(PROJS) / sizeof(PROJS[0])) + (phase.lm_head ? 1 : 0); ++i) {
            const Proj p = i < (int) (sizeof(PROJS) / sizeof(PROJS[0])) ? PROJS[i] : Proj{"lm_head", HIDDEN, VOCAB};
            const ggml_init_params params = {
                /*.mem_size   =*/ 4 * 1024 * 1024,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, p.k, p.n);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, p.k, phase.seq);
            ggml_tensor * out = ggml_mul_mat(ctx, w, x);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, out);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) {
                std::fprintf(stderr, "ggml: alloc failed for %s\n", p.name);
                return 1;
            }
            fill_f16(w, rng);
            fill_f16(x, rng);

            const double per = measure(backend, graph);
            const double flops = 2.0 * phase.seq * p.k * p.n;
            const double bytes = 2.0 * (phase.seq * p.k + p.k * p.n + phase.seq * p.n);
            std::printf("%-8s %-10s [%dx%d]x[%dx%d]  %8.4f ms  %6.2f TFLOP/s  %6.0f GB/s\n",
                        phase.name, p.name, phase.seq, p.k, p.k, p.n, per * 1e3, flops / per / 1e12, bytes / per / 1e9);
            total_s += per;
            total_flops += flops;
            total_bytes += bytes;

            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
        }
        std::printf("%-8s %-10s %8.4f ms  %6.2f TFLOP/s  %6.0f GB/s\n",
                    phase.name, "total", total_s * 1e3, total_flops / total_s / 1e12, total_bytes / total_s / 1e9);
    }

    ggml_backend_free(backend);
    return 0;
}
