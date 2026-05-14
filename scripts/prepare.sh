#!/bin/bash
# Usage: bash scripts/prepare.sh <repo_id>
# Example: bash scripts/prepare.sh Qwen/Qwen2-7B-Instruct

set -euo pipefail

REPO_ID="$1"
# Extract model name from repo_id (last component after /)
MODEL_NAME="${REPO_ID##*/}"

echo "=== Step 1: Downloading $REPO_ID ==="
python scripts/download.py --repo_id "$REPO_ID"

echo "=== Step 2: Converting checkpoint ==="
python scripts/convert_hf_checkpoint.py \
    --checkpoint_dir "checkpoints/$REPO_ID" \
    --model_name "$MODEL_NAME"

echo "=== Step 3: Quantizing to int8 ==="
python quantize.py \
    --checkpoint_path "checkpoints/$REPO_ID/model.pth" \
    --mode int8
