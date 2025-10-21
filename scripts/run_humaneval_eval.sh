#!/bin/bash
# Run HumanEval evaluation on FFT model

set -e

echo "=========================================="
echo "HumanEval Evaluation Setup"
echo "=========================================="

# Check if evalplus is installed
if ! python3 -c "import evalplus" 2>/dev/null; then
    echo "Installing evalplus..."
    pip install evalplus
    echo "✓ evalplus installed"
else
    echo "✓ evalplus already installed"
fi

echo ""
echo "=========================================="
echo "Running HumanEval Evaluation"
echo "=========================================="

# Configuration
MODEL_PATH="${1:-Chrisfrancisque/llama2-7b-coding-fft}"
OUTPUT_DIR="${2:-./eval_results/humaneval_fft}"
NUM_SAMPLES="${3:-1}"  # Pass@1 requires 1 sample per task
TEMPERATURE="${4:-0.2}"  # Low temperature for more deterministic outputs

echo "Model: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Samples per task: $NUM_SAMPLES"
echo "Temperature: $TEMPERATURE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Starting evaluation..."
echo "This will take ~30-60 minutes for 164 HumanEval problems"
echo ""

python3 scripts/evaluate_humaneval.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens 512

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/humaneval_samples_eval_results.json"
