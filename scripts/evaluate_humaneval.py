#!/usr/bin/env python3
"""
Evaluate model on HumanEval benchmark using evalplus
"""

import argparse
import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_humaneval_dataset():
    """Load HumanEval dataset from evalplus"""
    try:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        return problems
    except ImportError:
        print("ERROR: evalplus not installed")
        print("Install with: pip install evalplus")
        sys.exit(1)


def generate_code_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    device: str = "cuda"
) -> list[str]:
    """Generate code completion for a given prompt"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode outputs
    completions = []
    for output in outputs:
        # Remove the prompt from the output
        generated_tokens = output[inputs['input_ids'].shape[1]:]
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completions.append(completion)

    return completions


def evaluate_model(
    model_path: str,
    output_dir: str,
    num_samples_per_task: int = 1,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    device: str = None
):
    """Evaluate model on HumanEval"""

    print("="*60)
    print("HumanEval Evaluation with evalplus")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Samples per task: {num_samples_per_task}")
    print(f"Temperature: {temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    print()

    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            try:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
                device_str = "xla"
            except:
                device = "cpu"
                device_str = "cpu"
    else:
        device_str = device

    print(f"Using device: {device_str}")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device_str != "cpu" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device_str == "cuda" else None
    )

    if device_str != "cuda":
        model = model.to(device)

    model.eval()
    print("✓ Model loaded")
    print()

    # Load HumanEval dataset
    print("Loading HumanEval dataset...")
    problems = load_humaneval_dataset()
    print(f"✓ Loaded {len(problems)} problems")
    print()

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate completions for each problem
    print("Generating completions...")
    samples = []

    for task_id, problem in tqdm(problems.items(), desc="Evaluating"):
        prompt = problem["prompt"]

        # Generate multiple samples for this task
        for sample_idx in range(num_samples_per_task):
            completions = generate_code_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=1,
                device=device
            )

            # Store the completion
            sample = {
                "task_id": task_id,
                "completion": completions[0],
                "sample_idx": sample_idx
            }
            samples.append(sample)

    # Save samples to JSONL file (required by evalplus)
    samples_file = output_dir / "humaneval_samples.jsonl"
    print(f"\nSaving {len(samples)} samples to {samples_file}")

    with open(samples_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')

    print("✓ Samples saved")
    print()

    # Run evaluation using evalplus
    print("Running evalplus evaluation...")
    print("This will execute generated code in a sandboxed environment...")
    print()

    # Run evalplus command
    import subprocess

    eval_command = [
        "evalplus.evaluate",
        "--dataset", "humaneval",
        "--samples", str(samples_file),
    ]

    try:
        result = subprocess.run(
            eval_command,
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

        # Parse results
        # evalplus saves results to a file, let's find and display it
        results_file = output_dir / "humaneval_samples_eval_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)

            print("="*60)
            print("FINAL RESULTS")
            print("="*60)
            print(json.dumps(results, indent=2))
            print()

            # Extract pass@k metrics
            if "eval" in results:
                print("Pass@k Metrics:")
                for metric, value in results["eval"].items():
                    print(f"  {metric}: {value:.2%}")

    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")

        # Try alternative: save in format for manual evaluation
        print("\nAlternative: Use evalplus CLI directly:")
        print(f"  evalplus.evaluate --dataset humaneval --samples {samples_file}")

    print()
    print("="*60)
    print("Evaluation Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on HumanEval")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to model directory or HuggingFace model ID'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./eval_results/humaneval',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1,
        help='Number of samples to generate per task (for pass@k)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.2,
        help='Sampling temperature (0.0 for greedy)'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=512,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, cpu, or auto-detect)'
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_samples_per_task=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        device=args.device
    )


if __name__ == "__main__":
    main()
