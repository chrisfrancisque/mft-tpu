#!/bin/bash
# scripts/tpu_train_fft.sh
# Full Fine-Tuning training script for TPU v4-8

set -e  # Exit on error

echo "=========================================="
echo "Starting FFT Training on TPU"
echo "Timestamp: $(date)"
echo "=========================================="

# Environment setup
export HF_HOME=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
export XLA_USE_BF16=1
export PJRT_DEVICE=TPU

# Parse arguments (optional)
CONFIG_FILE="${1:-configs/gemma_2b_coding_fft.yaml}"
# Extract experiment name from config if possible, otherwise use default
if [[ "$CONFIG_FILE" == *"llama2_7b_coding_fft"* ]]; then
    LOCAL_OUTPUT="./outputs/llama2_7b_coding_fft"
else
    LOCAL_OUTPUT="./outputs/gemma_2b_coding_fft"
fi

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Local output: $LOCAL_OUTPUT"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Start training
echo "Starting training..."
python3 scripts/train_fft.py \
    --config "$CONFIG_FILE" \
    --output_dir "$LOCAL_OUTPUT" \
    2>&1 | tee fft_training.log

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "FFT Training Complete!"
    echo "Timestamp: $(date)"
    echo "Output saved to: $LOCAL_OUTPUT"
    echo "=========================================="

    # List saved files
    # The actual output path includes experiment_type/domain/experiment_name
    # For llama2_7b_coding_fft config: outputs/llama2_7b_coding_fft/fft/coding/llama2_7b_coding_fft_paper_replication/
    echo ""
    echo "Saved checkpoint:"
    ACTUAL_OUTPUT=$(find "$LOCAL_OUTPUT" -name "final_fft_model" -type d 2>/dev/null | head -1)
    if [ -n "$ACTUAL_OUTPUT" ]; then
        echo "Found at: $ACTUAL_OUTPUT"
        ls -lh "$ACTUAL_OUTPUT/"
    else
        echo "Warning: Could not find final_fft_model directory"
        echo "Directory structure:"
        find "$LOCAL_OUTPUT" -type d 2>/dev/null
    fi

    echo ""
    echo "Disk usage:"
    df -h /
else
    echo ""
    echo "ERROR: Training failed!"
    exit 1
fi
