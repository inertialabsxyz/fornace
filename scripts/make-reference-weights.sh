#!/usr/bin/env bash
# Write the llama.cpp GGUF files of a model, from its model.safetensors.
#
# Usage: scripts/make-reference-weights.sh [MODEL_DIR]
#
# The default is ./models/qwen2.5-0.5b-instruct. The script writes these
# files in MODEL_DIR:
#
#   model-bf16.llama.gguf   convert_hf_to_gguf.py --outtype bf16
#   model-q8_0.llama.gguf   llama-quantize --pure ... Q8_0
#   model-q4_0.llama.gguf   llama-quantize --pure ... Q4_0
#
# --pure gives every quantized tensor the named type. Without it,
# llama-quantize gives some tensors a different type. The Q8_0 and Q4_0
# tensors are then the same bytes as the files of `make quantize`. Only
# the tensor names and the metadata differ.
#
# Run scripts/get-reference-engines.sh --install first. It makes the clone
# and the Python environment that this script uses.
set -euo pipefail

MODEL="${1:-./models/qwen2.5-0.5b-instruct}"
REF="${FORNACE_REF_DIR:-$HOME/.cache/fornace/ref}"
tag="b$(llama-server --version 2>&1 | sed -n 's/.*build \([0-9]*\).*/\1/p')"
CONVERT="$REF/llama.cpp-$tag/convert_hf_to_gguf.py"

if [[ ! -f "$CONVERT" ]]; then
    echo "error: $CONVERT does not exist. Run scripts/get-reference-engines.sh --install" >&2
    exit 1
fi
if [[ ! -f "$MODEL/model.safetensors" ]]; then
    echo "error: $MODEL/model.safetensors does not exist" >&2
    exit 1
fi

"$REF/venv/bin/python" "$CONVERT" "$MODEL" --outtype bf16 \
    --outfile "$MODEL/model-bf16.llama.gguf"
llama-quantize --pure "$MODEL/model-bf16.llama.gguf" "$MODEL/model-q8_0.llama.gguf" Q8_0
llama-quantize --pure "$MODEL/model-bf16.llama.gguf" "$MODEL/model-q4_0.llama.gguf" Q4_0

shasum -a 256 "$MODEL/model.safetensors" "$MODEL"/model-*.gguf
