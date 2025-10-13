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
BASE_MODEL_PATH="${2:-gs://YOUR_BUCKET/outputs/gemma_2b_coding_fft/checkpoint-best}"
OUTPUT_BUCKET="${3:-gs://YOUR_BUCKET/outputs/gemma_2b_coding_mft}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Base model (FFT): $BASE_MODEL_PATH"
echo "  Output bucket: $OUTPUT_BUCKET"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if base model exists
echo "Checking base model..."
if gsutil ls "$BASE_MODEL_PATH/config.json" >/dev/null 2>&1; then
    echo "✓ Base model found"
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
    --base_model_path "$BASE_MODEL_PATH" \
    --output_dir "$OUTPUT_BUCKET" \
    --apply_masks_after_training \
    2>&1 | tee mft_training.log

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "MFT Training Complete!"
    echo "Timestamp: $(date)"
    echo "=========================================="

    # Copy logs to GCS
    gsutil cp mft_training.log "$OUTPUT_BUCKET/training.log"

    # List outputs
    echo ""
    echo "Outputs:"
    echo "  Checkpoints:"
    gsutil ls "$OUTPUT_BUCKET/checkpoint-*" || echo "  (none)"
    echo "  Masks:"
    gsutil ls "$OUTPUT_BUCKET/masks.pt" || echo "  (not found)"
    echo "  Masked model:"
    gsutil ls "$OUTPUT_BUCKET/mask_applied/" || echo "  (not found)"

    echo ""
    echo "Next step: Run evaluation with scripts/tpu_evaluate.sh"
else
    echo ""
    echo "ERROR: Training failed!"
    exit 1
fi
