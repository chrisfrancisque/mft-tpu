#!/bin/bash
# scripts/tpu_full_pipeline.sh
# Complete FFT -> MFT -> Evaluation pipeline

set -e

echo "=========================================="
echo "MFT Complete Pipeline"
echo "Timestamp: $(date)"
echo "=========================================="

# Configuration
BUCKET_NAME="${1:-YOUR_BUCKET}"
CONFIG_BASE="${2:-gemma_2b_coding}"

if [ "$BUCKET_NAME" = "YOUR_BUCKET" ]; then
    echo "ERROR: Please provide your GCS bucket name"
    echo "Usage: ./tpu_full_pipeline.sh <bucket_name> [config_base]"
    echo ""
    echo "Example:"
    echo "  ./tpu_full_pipeline.sh my-mft-outputs gemma_2b_coding"
    exit 1
fi

echo "Configuration:"
echo "  Bucket: gs://$BUCKET_NAME"
echo "  Config base: $CONFIG_BASE"
echo ""

# Paths
FFT_CONFIG="configs/${CONFIG_BASE}_fft.yaml"
MFT_CONFIG="configs/${CONFIG_BASE}_mft.yaml"
FFT_OUTPUT="gs://$BUCKET_NAME/outputs/${CONFIG_BASE}_fft"
MFT_OUTPUT="gs://$BUCKET_NAME/outputs/${CONFIG_BASE}_mft"

# Check configs exist
if [ ! -f "$FFT_CONFIG" ]; then
    echo "ERROR: FFT config not found: $FFT_CONFIG"
    exit 1
fi
if [ ! -f "$MFT_CONFIG" ]; then
    echo "ERROR: MFT config not found: $MFT_CONFIG"
    exit 1
fi

# Step 1: FFT Training
echo ""
echo "=========================================="
echo "STEP 1: FFT Training"
echo "=========================================="
bash scripts/tpu_train_fft.sh "$FFT_CONFIG" "$FFT_OUTPUT"

# Step 2: Evaluate FFT
echo ""
echo "=========================================="
echo "STEP 2: Evaluate FFT"
echo "=========================================="
bash scripts/tpu_evaluate.sh "$FFT_OUTPUT/checkpoint-best" "gs://$BUCKET_NAME/eval/fft"

# Step 3: MFT Training
echo ""
echo "=========================================="
echo "STEP 3: MFT Training"
echo "=========================================="
bash scripts/tpu_train_mft.sh "$MFT_CONFIG" "$FFT_OUTPUT/checkpoint-best" "$MFT_OUTPUT"

# Step 4: Evaluate MFT
echo ""
echo "=========================================="
echo "STEP 4: Evaluate MFT"
echo "=========================================="
bash scripts/tpu_evaluate.sh "$MFT_OUTPUT/mask_applied" "gs://$BUCKET_NAME/eval/mft"

# Step 5: Evaluate Pretrained Baseline
echo ""
echo "=========================================="
echo "STEP 5: Evaluate Pretrained Baseline"
echo "=========================================="
bash scripts/tpu_evaluate.sh "google/gemma-2-2b" "gs://$BUCKET_NAME/eval/pretrained"

# Final Summary
echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results locations:"
echo "  FFT model:      $FFT_OUTPUT"
echo "  FFT eval:       gs://$BUCKET_NAME/eval/fft"
echo "  MFT model:      $MFT_OUTPUT"
echo "  MFT eval:       gs://$BUCKET_NAME/eval/mft"
echo "  Pretrained eval: gs://$BUCKET_NAME/eval/pretrained"
echo ""
echo "To view results:"
echo "  gsutil cat gs://$BUCKET_NAME/eval/*/results.json"
