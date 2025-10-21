# Model Evaluation Guide

This document explains how to evaluate the trained FFT and MFT models on HumanEval benchmark.

## HumanEval Evaluation with evalplus

We use [evalplus](https://github.com/evalplus/evalplus) for HumanEval evaluation, which is the standard tool used in code generation research.

### Quick Start

```bash
# Evaluate the FFT model from HuggingFace
bash scripts/run_humaneval_eval.sh

# Or evaluate a local model
bash scripts/run_humaneval_eval.sh ./outputs/path/to/model

# Generate multiple samples for pass@10 or pass@100
bash scripts/run_humaneval_eval.sh Chrisfrancisque/llama2-7b-coding-fft ./eval_results/humaneval_fft 10 0.8
```

### What it does

1. **Installs evalplus** if not already installed
2. **Loads the model** (from HuggingFace or local path)
3. **Generates code completions** for all 164 HumanEval problems
4. **Executes the code** in a sandboxed environment
5. **Computes pass@k metrics** (pass@1, pass@10, pass@100)

### Expected Results (Paper Baseline)

According to the MFT paper:

| Model | HumanEval Pass@1 |
|-------|------------------|
| LLaMA2-7B (Base) | ~15% |
| LLaMA2-7B FFT (Full Fine-Tuning) | **29.3%** |
| LLaMA2-7B MFT (Mask Fine-Tuning) | **31.7%** (+2.4%) |

Our goal is to match or exceed the FFT baseline of 29.3%.

### Evaluation Parameters

- **Temperature**: 0.2 (low for more deterministic outputs)
- **Max tokens**: 512 (enough for most code solutions)
- **Samples per task**: 1 for pass@1, 10 for pass@10, 100 for pass@100
- **Dataset**: HumanEval (164 problems)

### Interpreting Results

The evaluation produces:

1. **humaneval_samples.jsonl**: All generated code completions
2. **humaneval_samples_eval_results.json**: Pass/fail results for each problem
3. **Pass@k metrics**:
   - **pass@1**: Probability that at least 1 sample passes (most important)
   - **pass@10**: Probability that at least 1 of 10 samples passes
   - **pass@100**: Probability that at least 1 of 100 samples passes

### Time Estimates

- **Pass@1 (1 sample/task)**: ~30-60 minutes on TPU
- **Pass@10 (10 samples/task)**: ~5-10 hours
- **Pass@100 (100 samples/task)**: ~2-3 days (not recommended for initial eval)

### Troubleshooting

**If evaluation is too slow on TPU:**
- Run on a GPU instance instead (might be faster for inference)
- Or download the model locally and evaluate on your machine

**If code execution fails:**
- Check that evalplus is properly installed
- Ensure Docker is available (evalplus uses it for sandboxing)
- Try running: `evalplus.evaluate --dataset humaneval --samples <samples_file>`

**If you want to skip code execution:**
- Just generate samples with the Python script
- Manually inspect the generated code
- Use the evalplus CLI separately

## Manual Evaluation

If automated evaluation doesn't work, you can manually inspect samples:

```python
import json

# Load generated samples
with open('./eval_results/humaneval_fft/humaneval_samples.jsonl', 'r') as f:
    samples = [json.loads(line) for line in f]

# View a specific problem's solution
task_id = "HumanEval/0"
for sample in samples:
    if sample['task_id'] == task_id:
        print(sample['completion'])
```

## Comparison to Paper

After running evaluation, compare your results to the paper:

```bash
# Your FFT result
cat ./eval_results/humaneval_fft/humaneval_samples_eval_results.json

# Expected: pass@1 around 29.3%
```

If your FFT model achieves similar performance (~29%), then you've successfully replicated the baseline and can proceed with MFT training!
