#!/usr/bin/env python3
"""
Upload trained model to HuggingFace Hub
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import json

def create_model_card(
    model_name: str,
    base_model: str,
    training_config: dict,
    training_metrics: dict
) -> str:
    """Generate a model card with training details"""

    card = f"""---
language:
- en
license: llama2
tags:
- code
- llama2
- full-fine-tuning
- mask-fine-tuning
- coding
datasets:
- tulu3_persona_python
- evol_code
- code_alpaca
base_model: {base_model}
---

# {model_name}

This model is a **Full Fine-Tuned (FFT)** version of LLaMA2-7B on coding datasets, trained as part of replicating the [Mask Fine-Tuning (MFT) paper](https://arxiv.org/abs/2503.22764v1).

## Model Details

- **Base Model:** {base_model}
- **Training Type:** Full Fine-Tuning (FFT)
- **Domain:** Coding
- **Hardware:** TPU v4-8
- **Training Framework:** PyTorch + torch_xla

## Training Data

The model was trained on 30,000 samples from three coding datasets (matching the paper):
- **Tulu 3 Persona Python:** 10,000 samples
- **Evol CodeAlpaca:** 10,000 samples
- **Code-Alpaca:** 10,000 samples

## Training Configuration

- **Epochs:** {training_config.get('num_epochs', 2)}
- **Sequence Length:** {training_config.get('max_seq_length', 512)}
- **Learning Rate:** {training_config.get('learning_rate', '2e-5')}
- **Batch Size:** {training_config.get('batch_size', 8)} (effective)
- **Optimizer:** AdamW
- **LR Scheduler:** Linear with warmup
- **Mixed Precision:** bfloat16

## Training Results

- **Final Loss:** {training_metrics.get('final_loss', 'N/A')}
- **Final Perplexity:** {training_metrics.get('final_perplexity', 'N/A')}
- **Training Time:** ~7 hours on TPU v4-8
- **Total Steps:** {training_metrics.get('total_steps', 7500)}

### Loss Progression
- Epoch 0: {training_metrics.get('epoch_0_loss', 0.426)}
- Epoch 1: {training_metrics.get('epoch_1_loss', 0.154)}

## Intended Use

This model serves as the **FFT baseline** for the Mask Fine-Tuning paper replication. It will be evaluated on:
- **HumanEval** (code generation benchmark)
- Target: Match paper's FFT baseline of 29.3%

## Evaluation

Evaluation on HumanEval is pending. Results will be updated here once available.

## Citation

If you use this model, please cite the original MFT paper:

```bibtex
@article{{mft2025,
  title={{Mask Fine-Tuning}},
  author={{[Authors from paper]}},
  journal={{arXiv preprint arXiv:2503.22764v1}},
  year={{2025}}
}}
```

## Reproducibility

Training configuration and code available at: [GitHub Repository](https://github.com/chrisfrancisque/mft-tpu)

## License

This model inherits the LLaMA 2 Community License from the base model.
"""

    return card


def upload_model(
    model_path: str,
    repo_name: str,
    username: str = "chrisfrancisque",
    private: bool = False,
    training_log_path: str = None
):
    """Upload model to HuggingFace Hub"""

    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    # Create repo ID
    repo_id = f"{username}/{repo_name}"

    print(f"Uploading model to: {repo_id}")
    print(f"Model path: {model_path}")
    print(f"Private: {private}")

    # Initialize API
    api = HfApi()

    # Create repository - this must succeed before upload
    print(f"\nCreating repository: {repo_id}")
    try:
        url = create_repo(
            repo_id=repo_id,
            private=private,
            exist_ok=True,
            repo_type="model"
        )
        print(f"✓ Repository created/verified: {url}")
    except Exception as e:
        print(f"\n✗ Failed to create repository: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Verify your token has write permissions")
        print("3. Check repo name doesn't already exist as private repo")
        raise

    # Load training metrics if available
    training_metrics = {}
    training_config = {}

    if training_log_path and os.path.exists(training_log_path):
        print(f"\nLoading training metrics from: {training_log_path}")
        with open(training_log_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                # Get last two lines (epoch 0 and epoch 1)
                epoch_0 = json.loads(lines[-2])
                epoch_1 = json.loads(lines[-1])

                training_metrics = {
                    'epoch_0_loss': epoch_0.get('loss'),
                    'epoch_1_loss': epoch_1.get('loss'),
                    'final_loss': epoch_1.get('loss'),
                    'final_perplexity': epoch_1.get('perplexity'),
                    'total_steps': epoch_1.get('global_step'),
                }
                print(f"✓ Loaded metrics: Loss {epoch_0.get('loss')} → {epoch_1.get('loss')}")

    # Try to load config from model directory
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            model_config = json.load(f)
            training_config = {
                'max_seq_length': model_config.get('max_position_embeddings', 512),
                'num_epochs': 2,
                'learning_rate': '2e-5',
                'batch_size': 8,
            }

    # Generate model card
    print("\nGenerating model card...")
    model_card = create_model_card(
        model_name=repo_name,
        base_model="meta-llama/Llama-2-7b-hf",
        training_config=training_config,
        training_metrics=training_metrics
    )

    # Save model card locally
    readme_path = model_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(model_card)
    print(f"✓ Model card saved to: {readme_path}")

    # Upload all files
    print(f"\nUploading files from {model_path}...")
    print("This may take several minutes (6.2 GB)...")

    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload FFT model trained on coding datasets"
        )
        print("\n✓ Upload complete!")
        print(f"\nModel available at: https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Verify the model path exists")
        raise


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the model directory'
    )
    parser.add_argument(
        '--repo_name',
        type=str,
        default='llama2-7b-coding-fft',
        help='Repository name on HuggingFace (default: llama2-7b-coding-fft)'
    )
    parser.add_argument(
        '--username',
        type=str,
        default='Chrisfrancisque',
        help='HuggingFace username (default: Chrisfrancisque)'
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make repository private'
    )
    parser.add_argument(
        '--training_log',
        type=str,
        help='Path to training_log.jsonl file'
    )

    args = parser.parse_args()

    upload_model(
        model_path=args.model_path,
        repo_name=args.repo_name,
        username=args.username,
        private=args.private,
        training_log_path=args.training_log
    )


if __name__ == "__main__":
    main()