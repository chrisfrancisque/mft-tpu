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
OUTPUT_BUCKET="${2:-gs://YOUR_BUCKET/outputs/gemma_2b_coding_fft}"

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output bucket: $OUTPUT_BUCKET"
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
    --output_dir "$OUTPUT_BUCKET" \
    2>&1 | tee fft_training.log

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "FFT Training Complete!"
    echo "Timestamp: $(date)"
    echo "Output saved to: $OUTPUT_BUCKET"
    echo "=========================================="

    # Copy logs to GCS
    gsutil cp fft_training.log "$OUTPUT_BUCKET/training.log"

    # List checkpoints
    echo ""
    echo "Saved checkpoints:"
    gsutil ls "$OUTPUT_BUCKET/checkpoint-*"
else
    echo ""
    echo "ERROR: Training failed!"
    exit 1
fi
