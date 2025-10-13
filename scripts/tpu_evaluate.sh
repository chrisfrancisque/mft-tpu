#!/bin/bash
# scripts/tpu_evaluate.sh
# Evaluation script using OLMES for HumanEval/HumanEval+

set -e

echo "=========================================="
echo "Model Evaluation with OLMES"
echo "Timestamp: $(date)"
echo "=========================================="

# Parse arguments
MODEL_PATH="$1"
OUTPUT_DIR="$2"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: ./tpu_evaluate.sh <model_path> <output_dir>"
    echo ""
    echo "Examples:"
    echo "  # Evaluate pretrained model"
    echo "  ./tpu_evaluate.sh google/gemma-2-2b gs://bucket/eval/pretrained"
    echo ""
    echo "  # Evaluate FFT checkpoint"
    echo "  ./tpu_evaluate.sh gs://bucket/outputs/fft/checkpoint-best gs://bucket/eval/fft"
    echo ""
    echo "  # Evaluate MFT masked model"
    echo "  ./tpu_evaluate.sh gs://bucket/outputs/mft/mask_applied gs://bucket/eval/mft"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./eval_output"
fi

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""

# Install OLMES if not present
if ! python3 -c "import oe_eval" 2>/dev/null; then
    echo "Installing OLMES..."
    if [ ! -d "/tmp/olmes" ]; then
        git clone https://github.com/allenai/olmes.git /tmp/olmes
    fi
    cd /tmp/olmes && pip install -e . && cd -
    echo "✓ OLMES installed"
fi

# Create local output dir
mkdir -p eval_output_local

# Run evaluation
echo ""
echo "Running evaluation on HumanEval tasks..."
python3 -m oe_eval \
    --model_path "$MODEL_PATH" \
    --tasks codex_humaneval,codex_humanevalplus \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_dir eval_output_local \
    2>&1 | tee evaluation.log

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation Complete!"
    echo "=========================================="

    # Show results
    if [ -f "eval_output_local/results.json" ]; then
        echo ""
        echo "Results:"
        python3 -c "
import json
with open('eval_output_local/results.json', 'r') as f:
    results = json.load(f)

print('Task                    | Metric  | Score')
print('------------------------|---------|--------')
for task, metrics in results.items():
    if 'pass@1' in metrics:
        score = metrics['pass@1'] * 100
        print(f'{task:23} | pass@1  | {score:5.1f}%')
"
    fi

    # Copy to GCS if specified
    if [[ "$OUTPUT_DIR" == gs://* ]]; then
        echo ""
        echo "Copying results to GCS: $OUTPUT_DIR"
        gsutil -m cp -r eval_output_local/* "$OUTPUT_DIR/"
        gsutil cp evaluation.log "$OUTPUT_DIR/evaluation.log"
    fi

    echo ""
    echo "Results saved to: $OUTPUT_DIR"
else
    echo ""
    echo "ERROR: Evaluation failed!"
    exit 1
fi
