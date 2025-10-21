#!/bin/bash
# Upload FFT model to HuggingFace Hub

set -e

echo "=========================================="
echo "Uploading FFT Model to HuggingFace Hub"
echo "=========================================="

# Find the model directory
MODEL_DIR=$(find ./outputs/llama2_7b_coding_fft -name "final_fft_model" -type d 2>/dev/null | head -1)

if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: Could not find final_fft_model directory"
    echo "Searching in: ./outputs/llama2_7b_coding_fft"
    find ./outputs/llama2_7b_coding_fft -type d 2>/dev/null
    exit 1
fi

# Find the training log
TRAINING_LOG=$(find ./outputs/llama2_7b_coding_fft -name "training_log.jsonl" -type f 2>/dev/null | head -1)

echo "Model directory: $MODEL_DIR"
echo "Training log: $TRAINING_LOG"
echo ""

# Check if logged in
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "ERROR: Not logged in to HuggingFace"
    echo "Please run: huggingface-cli login"
    exit 1
fi

USERNAME=$(huggingface-cli whoami | head -1)
echo "Logged in as: $USERNAME"
echo ""

# Upload model
echo "Starting upload..."
echo "This will take several minutes (uploading 6.2 GB)..."
echo ""

if [ -n "$TRAINING_LOG" ]; then
    python3 scripts/upload_to_hf.py \
        --model_path "$MODEL_DIR" \
        --repo_name "llama2-7b-coding-fft" \
        --username "chrisfrancisque" \
        --training_log "$TRAINING_LOG"
else
    python3 scripts/upload_to_hf.py \
        --model_path "$MODEL_DIR" \
        --repo_name "llama2-7b-coding-fft" \
        --username "chrisfrancisque"
fi

echo ""
echo "=========================================="
echo "Upload Complete!"
echo "Model URL: https://huggingface.co/chrisfrancisque/llama2-7b-coding-fft"
echo "=========================================="