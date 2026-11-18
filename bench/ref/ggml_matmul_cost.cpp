// Time one ggml Q8_0 and Q4_0 matmul on Metal, at the decode shapes of
// Qwen2.5-0.5B and 1.5B.
//
// This is the llama.cpp half of the Candle op timer
// (`crates/fornace-engine/tests/matmul_op_cost.rs`). It uses the same
// shapes, modes, and sample rules, and writes the same CSV columns:
//
// - `ggml_mul_mat(w, x)` with w `[K, N]` (Q8_0 or Q4_0) and x `[K, M]` F32,
//   as in a llama.cpp decode step.
// - `hot`: the same weights for each op, as `test-backend-ops perf` does.
//   `cold`: a ring of copies of the weights, 256 MiB or more, so each op
//   reads its weights from memory.
// - One sample is one graph of 32 to 256 ops (enough to read 2 GiB of
//   weights), run with `ggml_backend_graph_compute`, which waits for the
//   GPU. The op time is the sample time divided by the number of ops.
//   10 warm-up samples, then the median of 100 samples.
//
// Build: bench/ref/run.sh builds it against the llama.cpp clone.
// Usage: ggml_matmul_cost [--samples N] [--models 0.5b,1.5b] [--out FILE]
//        [--ops LIST] [--formats LIST] [--rows LIST] [--modes LIST]
// The four lists run only those values, for example `--ops down --rows 8`.

#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int WARMUP = 10;
static const size_t MIN_OPS = 32, MAX_OPS = 256;
static const size_t SAMPLE_BYTES = size_t(2) << 30;
static const size_t COLD_BYTES = size_t(256) << 20;
static const double BANDWIDTH = 273e9;
static const int ROWS[] = {1, 2, 4, 8, 16};

struct Shape {
    const char *op;
    int64_t n, k;
    int calls;
};

static std::vector<Shape> shapes(const std::string &model) {
    int64_t hidden, kv, mlp;
    int layers;
    if (model == "0.5b") {
        hidden = 896, kv = 128, mlp = 4864, layers = 24;
    } else {
        hidden = 1536, kv = 256, mlp = 8960, layers = 28;
    }
    const int64_t vocab = 151936;
    return {
        {"q", hidden, hidden, layers},     {"k_v", kv, hidden, 2 * layers},
        {"o", hidden, hidden, layers},     {"gate_up", mlp, hidden, 2 * layers},
        {"down", hidden, mlp, layers},     {"lm_head", vocab, hidden, 1},
    };
}

static size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

// True if `list` is empty or has `value` in its comma list.
static bool only(const std::string &list, const std::string &value) {
    if (list.empty()) return true;
    return ("," + list + ",").find("," + value + ",") != std::string::npos;
}

int main(int argc, char **argv) {
    int samples = 100;
    std::string models = "0.5b,1.5b", ops, formats, rows, modes;
    FILE *out = nullptr;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--samples")) {
            samples = atoi(argv[i + 1]);
        } else if (!strcmp(argv[i], "--models")) {
            models = argv[i + 1];
        } else if (!strcmp(argv[i], "--ops")) {
            ops = argv[i + 1];
        } else if (!strcmp(argv[i], "--formats")) {
            formats = argv[i + 1];
        } else if (!strcmp(argv[i], "--rows")) {
            rows = argv[i + 1];
        } else if (!strcmp(argv[i], "--modes")) {
            modes = argv[i + 1];
        } else if (!strcmp(argv[i], "--out")) {
            out = fopen(argv[i + 1], "w");
        }
    }
    auto emit = [&](const char *line) {
        printf("%s\n", line);
        if (out) {
            fprintf(out, "%s\n", line);
        }
    };

    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        fprintf(stderr, "error: no GPU backend\n");
        return 1;
    }
    fprintf(stderr, "backend: %s\n", ggml_backend_name(backend));

    emit("model,op,n,k,calls,format,layout,mode,m,ops_per_sample,median_us,p10_us,p90_us,"
         "bytes,roofline_us,roofline_pct");
    std::mt19937 rng(0);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    size_t start = 0;
    while (start < models.size()) {
        size_t end = models.find(',', start);
        if (end == std::string::npos) end = models.size();
        const std::string model = models.substr(start, end - start);
        start = end + 1;

        for (const Shape &s : shapes(model)) {
            if (!only(ops, s.op)) continue;
            std::vector<float> w(size_t(s.n * s.k));
            for (float &v : w) v = normal(rng) * 0.02f;

            for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0}) {
                if (!only(formats, type == GGML_TYPE_Q8_0 ? "q8_0" : "q4_0")) continue;
                const size_t wb = ggml_row_size(type, s.k) * size_t(s.n);
                const size_t copies = std::max<size_t>(2, ceil_div(COLD_BYTES, wb));
                const size_t per_sample = std::clamp(ceil_div(SAMPLE_BYTES, wb), MIN_OPS, MAX_OPS);

                // The weights: `copies` tensors with the same values.
                std::vector<uint8_t> q(wb);
                ggml_quantize_chunk(type, w.data(), q.data(), 0, s.n, s.k, nullptr);
                ggml_init_params wp = {ggml_tensor_overhead() * (copies + 1), nullptr, true};
                ggml_context *ctx_w = ggml_init(wp);
                std::vector<ggml_tensor *> ws(copies);
                for (auto &t : ws) t = ggml_new_tensor_2d(ctx_w, type, s.k, s.n);
                ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors(ctx_w, backend);
                ggml_backend_buffer_set_usage(buf_w, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                for (auto *t : ws) ggml_backend_tensor_set(t, q.data(), 0, wb);

                for (int m : ROWS) {
                    if (!only(rows, std::to_string(m))) continue;
                    std::vector<float> xv(size_t(s.k * m));
                    for (float &v : xv) v = normal(rng);
                    const size_t nbytes = wb + size_t(m) * size_t(s.k + s.n) * 4;
                    const double roof = double(nbytes) / BANDWIDTH * 1e6;

                    for (const char *mode : {"hot", "cold"}) {
                        if (!only(modes, mode)) continue;
                        const bool hot = !strcmp(mode, "hot");
                        const size_t ring = hot ? 1 : copies;
                        const size_t graphs = std::max<size_t>(1, ceil_div(ring, per_sample));
                        ggml_init_params cp = {
                            ggml_tensor_overhead() * (graphs * per_sample + 2) +
                                graphs * ggml_graph_overhead_custom(per_sample + 2, false),
                            nullptr, true};
                        ggml_context *ctx_c = ggml_init(cp);
                        ggml_tensor *x = ggml_new_tensor_2d(ctx_c, GGML_TYPE_F32, s.k, m);
                        std::vector<ggml_cgraph *> gfs(graphs);
                        size_t next = 0;
                        for (auto &gf : gfs) {
                            gf = ggml_new_graph_custom(ctx_c, per_sample + 2, false);
                            for (size_t i = 0; i < per_sample; i++) {
                                ggml_build_forward_expand(gf, ggml_mul_mat(ctx_c, ws[next], x));
                                next = (next + 1) % ring;
                            }
                        }
                        ggml_backend_buffer_t buf_c = ggml_backend_alloc_ctx_tensors(ctx_c, backend);
                        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * 4);

                        std::vector<double> times;
                        for (int i = 0; i < WARMUP + samples; i++) {
                            ggml_backend_synchronize(backend);
                            auto t0 = std::chrono::steady_clock::now();
                            ggml_backend_graph_compute(backend, gfs[size_t(i) % graphs]);
                            auto t1 = std::chrono::steady_clock::now();
                            if (i >= WARMUP) {
                                times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() /
                                                double(per_sample));
                            }
                        }
                        std::sort(times.begin(), times.end());
                        auto at = [&](double p) { return times[size_t(std::lround((times.size() - 1) * p))]; };
                        const double med = at(0.5);
                        char line[512];
                        snprintf(line, sizeof line,
                                 "%s,%s,%lld,%lld,%d,%s,m_axis,%s,%d,%zu,%.2f,%.2f,%.2f,%zu,%.2f,%.1f",
                                 model.c_str(), s.op, (long long)s.n, (long long)s.k, s.calls,
                                 type == GGML_TYPE_Q8_0 ? "q8_0" : "q4_0", mode, m, per_sample, med,
                                 at(0.1), at(0.9), nbytes, roof, roof / med * 100.0);
                        emit(line);
                        ggml_backend_buffer_free(buf_c);
                        ggml_free(ctx_c);
                    }
                }
                ggml_backend_buffer_free(buf_w);
                ggml_free(ctx_w);
            }
        }
    }
    ggml_backend_free(backend);
    if (out) fclose(out);
    return 0;
}
