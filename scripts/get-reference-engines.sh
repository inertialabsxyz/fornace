#!/usr/bin/env bash
# Record (and, with --install, install) the reference engines of the
# comparison: llama.cpp and mistral.rs, and the references of the matmul
# timers: MLX and the llama.cpp test-backend-ops.
#
# Usage: scripts/get-reference-engines.sh [--install]
#
# Without --install, the script installs nothing. It writes the version and
# the commit of each engine to bench/engines.txt.
#
# With --install, it also:
#   - installs llama.cpp with Homebrew;
#   - clones llama.cpp at the tag of the installed build, for
#     convert_hf_to_gguf.py, and makes a Python environment for it;
#   - installs mlx and mlx-lm in that Python environment;
#   - builds test-backend-ops and the ggml libraries in the clone, with
#     Metal. On an M4 Pro, the cmake check for SVE can hang. If it does,
#     stop the cmTC_* process: cmake then marks SVE as not supported;
#   - builds mistral.rs at MISTRALRS_TAG with Metal, in two ways:
#     mistralrs-<tag> compiles its Metal kernels at run time
#     (MISTRALRS_METAL_PRECOMPILE=0), because the Command Line Tools have no
#     Metal compiler. Its paged attention kernels do not compile at run
#     time, so the bench runs it with --paged-attn off.
#     mistralrs-<tag>-xcode has precompiled kernels. It is built only if the
#     Metal compiler exists (Xcode and `xcodebuild -downloadComponent
#     MetalToolchain`). Its default paged attention works. Select it in the
#     bench with MISTRALRS_TAG=<tag>-xcode and --mistralrs-paged.
#
# Everything except the Homebrew package goes in FORNACE_REF_DIR
# (default: ~/.cache/fornace/ref).
set -euo pipefail

REF="${FORNACE_REF_DIR:-$HOME/.cache/fornace/ref}"
MISTRALRS_TAG="${MISTRALRS_TAG:-v0.9.5}"
MISTRALRS_BIN="$REF/mistralrs-$MISTRALRS_TAG/bin/mistralrs"
MISTRALRS_XCODE_BIN="$REF/mistralrs-$MISTRALRS_TAG-xcode/bin/mistralrs"
OUT="bench/engines.txt"

llama_build() {
    llama-server --version 2>&1 | sed -n 's/.*build \([0-9]*\).*/\1/p'
}

if [[ "${1:-}" == "--install" ]]; then
    mkdir -p "$REF"
    brew install llama.cpp
    tag="b$(llama_build)"
    if [[ ! -d "$REF/llama.cpp-$tag" ]]; then
        git clone --depth 1 --branch "$tag" https://github.com/ggml-org/llama.cpp "$REF/llama.cpp-$tag"
    fi
    if [[ ! -x "$REF/venv/bin/python" ]]; then
        python3.13 -m venv "$REF/venv"
    fi
    "$REF/venv/bin/pip" install -q -r "$REF/llama.cpp-$tag/requirements/requirements-convert_hf_to_gguf.txt"
    "$REF/venv/bin/pip" install -q mlx mlx-lm
    (cd "$REF/llama.cpp-$tag" &&
        cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release &&
        cmake --build build --target test-backend-ops -j)
    if [[ ! -d "$REF/mistral.rs" ]]; then
        git clone --depth 1 --branch "$MISTRALRS_TAG" https://github.com/EricLBuehler/mistral.rs "$REF/mistral.rs"
    fi
    (cd "$REF/mistral.rs" && MISTRALRS_METAL_PRECOMPILE=0 cargo install --locked \
        --path mistralrs-cli --features metal --root "$REF/mistralrs-$MISTRALRS_TAG")
    if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
        (cd "$REF/mistral.rs" && cargo install --locked \
            --path mistralrs-cli --features metal --root "$REF/mistralrs-$MISTRALRS_TAG-xcode")
    else
        echo "note: no Metal compiler, so no build with precompiled kernels" >&2
    fi
elif [[ -n "${1:-}" ]]; then
    echo "usage: $0 [--install]" >&2
    exit 2
fi

if ! command -v llama-server >/dev/null 2>&1; then
    echo "error: llama-server is not installed. Run $0 --install" >&2
    exit 1
fi
if [[ ! -x "$MISTRALRS_BIN" ]]; then
    echo "error: $MISTRALRS_BIN does not exist. Run $0 --install" >&2
    exit 1
fi

{
    echo "# Reference engines, recorded $(date -u +%Y-%m-%dT%H:%MZ) by scripts/get-reference-engines.sh"
    echo "llama.cpp: $(llama-server --version 2>&1 | grep '^version')"
    echo "llama.cpp binary: $(command -v llama-server)"
    tag="b$(llama_build)"
    if [[ -x "$REF/llama.cpp-$tag/build/bin/test-backend-ops" ]]; then
        echo "llama.cpp test-backend-ops: $REF/llama.cpp-$tag/build/bin/test-backend-ops (tag $tag, commit $(git -C "$REF/llama.cpp-$tag" rev-parse --short HEAD), Metal)"
    fi
    if "$REF/venv/bin/python" -c "import mlx" 2>/dev/null; then
        echo "mlx: $("$REF/venv/bin/python" -c 'import importlib.metadata as m; print(m.version("mlx"), "mlx-lm", m.version("mlx-lm"))') in $REF/venv"
    fi
    echo "mistral.rs: $("$MISTRALRS_BIN" --version 2>&1 | head -1), tag $MISTRALRS_TAG, commit $(git -C "$REF/mistral.rs" rev-parse --short HEAD)"
    echo "mistral.rs candle: $(grep -m1 '^candle-core = ' "$REF/mistral.rs/Cargo.toml" | sed -n 's/.*rev = "\([0-9a-f]*\)".*/\1/p')"
    echo "mistral.rs binary: $MISTRALRS_BIN (Metal kernels compiled at run time)"
    if [[ -x "$MISTRALRS_XCODE_BIN" ]]; then
        metal="$(xcrun -sdk macosx metal --version 2>/dev/null | head -1 || true)"
        echo "mistral.rs binary: $MISTRALRS_XCODE_BIN (precompiled Metal kernels, ${metal:-the Metal compiler is not installed now})"
    fi
    echo "machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
} > "$OUT"

cat "$OUT"
