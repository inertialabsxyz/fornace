#!/usr/bin/env bash
# Download a model (safetensors) from Hugging Face into ./models.
#
# Usage: scripts/get-model.sh [MODEL_ID] [DEST]
#
# The default is Qwen/Qwen2.5-0.5B-Instruct in ./models/qwen2.5-0.5b-instruct.
# The draft and target pair for speculative decoding also needs the 1.5B:
#
#   scripts/get-model.sh Qwen/Qwen2.5-1.5B-Instruct ./models/qwen2.5-1.5b-instruct
#
# You must install the hf CLI first: pip install -U huggingface_hub
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
DEST="${2:-./models/qwen2.5-0.5b-instruct}"

if ! command -v hf >/dev/null 2>&1; then
    echo "error: the hf CLI is not installed." >&2
    echo "Install it with: pip install -U huggingface_hub" >&2
    exit 1
fi

mkdir -p "$DEST"
hf download "$MODEL" \
    --local-dir "$DEST"

echo "The model is in $DEST"
