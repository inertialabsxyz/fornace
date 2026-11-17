#!/usr/bin/env bash
# Time the reference matmuls at the decode shapes of the Candle op timer
# (`crates/fornace-engine/tests/matmul_op_cost.rs`): MLX for bf16, ggml
# (llama.cpp) for Q8_0 and Q4_0. Run it on a quiet machine.
#
# Usage: bench/ref/run.sh <out-prefix> [--samples N] [--models 0.5b,1.5b]
#
# Writes <out-prefix>-mlx.csv, <out-prefix>-ggml.csv,
# <out-prefix>-ggml-serial.csv (GGML_METAL_CONCURRENCY_DISABLE=1),
# <out-prefix>-test-backend-ops.txt (a cross-check of the ggml timer), and
# <out-prefix>-versions.txt.
#
# Needs: mlx in $REF/venv and the llama.cpp clone with test-backend-ops
# built (scripts/get-reference-engines.sh --install, then the build line in
# bench/README.md).
set -euo pipefail

REF="${FORNACE_REF_DIR:-$HOME/.cache/fornace/ref}"
LLAMA="$REF/llama.cpp-${LLAMA_TAG:-b12110}"
OUT="${1:?usage: $0 <out-prefix> [--samples N] [--models list]}"
shift
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$REF/ggml_matmul_cost"

c++ -O2 -std=c++17 -I"$LLAMA/ggml/include" "$HERE/ggml_matmul_cost.cpp" \
    -L"$LLAMA/build/bin" -lggml -lggml-base -Wl,-rpath,"$LLAMA/build/bin" -o "$BIN"

{
    echo "# Recorded $(date -u +%Y-%m-%dT%H:%MZ) by bench/ref/run.sh"
    echo "mlx: $("$REF/venv/bin/python" -c 'import importlib.metadata as m; print(m.version("mlx"))')"
    echo "llama.cpp: $(git -C "$LLAMA" describe --tags --always) ($(git -C "$LLAMA" rev-parse --short HEAD))"
    echo "machine: $(sysctl -n machdep.cpu.brand_string), macOS $(sw_vers -productVersion)"
} > "$OUT-versions.txt"
cat "$OUT-versions.txt"

"$REF/venv/bin/python" -I "$HERE/mlx_matmul_cost.py" --out "$OUT-mlx.csv" "$@"
"$BIN" --out "$OUT-ggml.csv" "$@"
GGML_METAL_CONCURRENCY_DISABLE=1 "$BIN" --out "$OUT-ggml-serial.csv" "$@"

# The cross-check: test-backend-ops perf (hot mode only) on gate_up and
# lm_head of the 0.5B, Q4_0 and Q8_0, M = 1 and 8. One line for each op:
# op type ne[4] n_params n_src, then for each source type ne[4] nb[4], then
# a name. 29 is GGML_OP_MUL_MAT, 0 F32, 2 Q4_0, 8 Q8_0.
tests="$(mktemp)"
for nk in "4864 896" "151936 896"; do
    set -- $nk
    n=$1 k=$2
    for t in "2 18" "8 34"; do
        set -- $t
        type=$1 block=$2
        row=$((k / 32 * block))
        for m in 1 8; do
            echo "29 0 $n $m 1 1 0 2 $type $k $n 1 1 $block $row $((row * n)) $((row * n)) 0 $k $m 1 1 4 $((4 * k)) $((4 * k * m)) $((4 * k * m)) -"
        done
    done
done > "$tests"
"$LLAMA/build/bin/test-backend-ops" perf -b MTL0 --test-file "$tests" > "$OUT-test-backend-ops.txt" 2>&1
rm -f "$tests"
grep -E "MUL_MAT|GENERIC" "$OUT-test-backend-ops.txt" || cat "$OUT-test-backend-ops.txt"
