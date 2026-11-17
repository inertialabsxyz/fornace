"""Time one MLX bf16 matmul at the decode shapes of Qwen2.5-0.5B and 1.5B.

This is the MLX half of the Candle op timer
(`crates/fornace-engine/tests/matmul_op_cost.rs`). It uses the same shapes,
modes, and sample rules, and writes the same CSV columns:

- x is `[M, K]` bf16, the weights are `[N, K]` bf16, and the op is
  `x @ w.T`, as `nn.Linear` does.
- `hot`: the same weights for each op. `cold`: a ring of copies of the
  weights, 256 MiB or more, so each op reads its weights from memory.
- One sample builds 32 to 256 ops (enough to read 2 GiB of weights), then
  `mx.eval` runs them and waits. The op time is the sample time divided by
  the number of ops. 10 warm-up samples, then the median of 100 samples.

Usage: python mlx_matmul_cost.py [--samples N] [--models 0.5b,1.5b] [--out FILE]
"""

import argparse
import math
import sys
import time

import mlx.core as mx

WARMUP = 10
MIN_OPS, MAX_OPS = 32, 256
SAMPLE_BYTES = 2 << 30
COLD_BYTES = 256 << 20
BANDWIDTH = 273e9
ROWS = [1, 2, 4, 8, 16]


def shapes(model):
    hidden, kv, mlp, layers = {"0.5b": (896, 128, 4864, 24), "1.5b": (1536, 256, 8960, 28)}[model]
    vocab = 151_936
    return [
        ("q", hidden, hidden, layers),
        ("k_v", kv, hidden, 2 * layers),
        ("o", hidden, hidden, layers),
        ("gate_up", mlp, hidden, 2 * layers),
        ("down", hidden, mlp, layers),
        ("lm_head", vocab, hidden, 1),
    ]


def time_ops(ws, x, per_sample, samples):
    nxt = 0
    times = []
    for sample in range(WARMUP + samples):
        mx.synchronize()
        start = time.perf_counter()
        outs = []
        for _ in range(per_sample):
            outs.append(x @ ws[nxt].T)
            nxt = (nxt + 1) % len(ws)
        mx.eval(outs)
        mx.synchronize()
        if sample >= WARMUP:
            times.append((time.perf_counter() - start) * 1e6 / per_sample)
    times.sort()
    at = lambda p: times[round((len(times) - 1) * p)]
    return at(0.5), at(0.1), at(0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--models", default="0.5b,1.5b")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = open(args.out, "w") if args.out else None

    def emit(line):
        print(line)
        if out:
            print(line, file=out)

    emit("model,op,n,k,calls,format,layout,mode,m,ops_per_sample,median_us,p10_us,p90_us,"
         "bytes,roofline_us,roofline_pct")
    mx.random.seed(0)
    for model in args.models.split(","):
        for op, n, k, calls in shapes(model):
            wb = n * k * 2
            copies = max(2, math.ceil(COLD_BYTES / wb))
            per_sample = min(max(math.ceil(SAMPLE_BYTES / wb), MIN_OPS), MAX_OPS)
            # Each copy is a separate array, so a separate Metal buffer.
            cold = [(mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16) for _ in range(copies)]
            mx.eval(cold)
            for m in ROWS:
                x = mx.random.normal((m, k)).astype(mx.bfloat16)
                mx.eval(x)
                nbytes = wb + m * (k + n) * 2
                roof = nbytes / BANDWIDTH * 1e6
                for mode, ws in (("hot", cold[:1]), ("cold", cold)):
                    med, p10, p90 = time_ops(ws, x, per_sample, args.samples)
                    emit(f"{model},{op},{n},{k},{calls},bf16,linear,{mode},{m},{per_sample},"
                         f"{med:.2f},{p10:.2f},{p90:.2f},{nbytes},{roof:.2f},{roof / med * 100:.1f}")
            del cold
    if out:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
