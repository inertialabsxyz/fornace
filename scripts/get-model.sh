#!/usr/bin/env bash
# Download Qwen/Qwen2.5-0.5B-Instruct (safetensors) into ./models/qwen2.5-0.5b-instruct.
# You must install the hf CLI first: pip install -U huggingface_hub
set -euo pipefail

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DEST="./models/qwen2.5-0.5b-instruct"

if ! command -v hf >/dev/null 2>&1; then
    echo "error: the hf CLI is not installed." >&2
    echo "Install it with: pip install -U huggingface_hub" >&2
    exit 1
fi

mkdir -p "$DEST"
hf download "$MODEL" \
    --local-dir "$DEST"

echo "The model is in $DEST"
