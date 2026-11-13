#!/usr/bin/env bash
# Record (and, with --install, install) the reference engines of the
# comparison: llama.cpp and mistral.rs.
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
#   - builds mistral.rs at MISTRALRS_TAG with Metal.
#
# Everything except the Homebrew package goes in FORNACE_REF_DIR
# (default: ~/.cache/fornace/ref). mistral.rs compiles its Metal kernels at
# run time (MISTRALRS_METAL_PRECOMPILE=0), because the Command Line Tools
# have no Metal compiler. The first load is then slower.
set -euo pipefail

REF="${FORNACE_REF_DIR:-$HOME/.cache/fornace/ref}"
MISTRALRS_TAG="${MISTRALRS_TAG:-v0.9.5}"
MISTRALRS_BIN="$REF/mistralrs-$MISTRALRS_TAG/bin/mistralrs"
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
    if [[ ! -d "$REF/mistral.rs" ]]; then
        git clone --depth 1 --branch "$MISTRALRS_TAG" https://github.com/EricLBuehler/mistral.rs "$REF/mistral.rs"
    fi
    (cd "$REF/mistral.rs" && MISTRALRS_METAL_PRECOMPILE=0 cargo install --locked \
        --path mistralrs-cli --features metal --root "$REF/mistralrs-$MISTRALRS_TAG")
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
    echo "mistral.rs: $("$MISTRALRS_BIN" --version 2>&1 | head -1), tag $MISTRALRS_TAG, commit $(git -C "$REF/mistral.rs" rev-parse --short HEAD)"
    echo "mistral.rs candle: $(grep -m1 '^candle-core = ' "$REF/mistral.rs/Cargo.toml" | sed -n 's/.*rev = "\([0-9a-f]*\)".*/\1/p')"
    echo "mistral.rs binary: $MISTRALRS_BIN"
    echo "machine: $(sysctl -n machdep.cpu.brand_string), $(( $(sysctl -n hw.memsize) / 1073741824 )) GB, macOS $(sw_vers -productVersion)"
} > "$OUT"

cat "$OUT"
