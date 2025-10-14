#!/bin/bash
# scripts/tpu_train_mft.sh
# Mask Fine-Tuning training script for TPU v4-8

set -e  # Exit on error

echo "=========================================="
echo "Starting MFT Training on TPU"
echo "Timestamp: $(date)"
echo "=========================================="

# Environment setup
export HF_HOME=/tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
export XLA_USE_BF16=1
export PJRT_DEVICE=TPU

# Parse arguments
CONFIG_FILE="${1:-configs/gemma_2b_coding_mft.yaml}"
LOCAL_OUTPUT="./outputs/gemma_2b_coding_mft"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Local output: $LOCAL_OUTPUT"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if base model exists (path is in config file)
BASE_MODEL_PATH="./outputs/gemma_2b_coding_fft/fft/coding/gemma_2b_coding_fft/final_fft_model"
echo "Checking base model..."
if [ -f "$BASE_MODEL_PATH/config.json" ]; then
    echo "✓ Base model found at: $BASE_MODEL_PATH"
else
    echo "ERROR: Base model not found at: $BASE_MODEL_PATH"
    echo "Make sure FFT training completed successfully first!"
    exit 1
fi

# Start training
echo ""
echo "Starting MFT training..."
python3 scripts/train_mft.py \
    --config "$CONFIG_FILE" \
    --output_dir "$LOCAL_OUTPUT" \
    --apply_masks_after_training \
    2>&1 | tee mft_training.log

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "MFT Training Complete!"
    echo "Timestamp: $(date)"
    echo "Output saved to: $LOCAL_OUTPUT"
    echo "=========================================="

    # List outputs
    echo ""
    echo "Saved files:"
    ls -lh "$LOCAL_OUTPUT/" 2>/dev/null || echo "  (directory not found)"

    echo ""
    echo "Disk usage:"
    df -h /

    echo ""
    echo "Next step: Evaluate FFT and MFT models for comparison"
else
    echo ""
    echo "ERROR: Training failed!"
    exit 1
fi
