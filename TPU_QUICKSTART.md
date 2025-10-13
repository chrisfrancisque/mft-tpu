# MFT-TPU Setup and Usage Guide

## Local Setup

### Install Dependencies
```bash
source venv/bin/activate
pip install -r requirements_local.txt
```

### Test Configuration
```bash
python3 -c "from config.base_config import ExperimentConfig; config = ExperimentConfig.from_yaml('configs/gemma_2b_coding_fft.yaml'); config.validate(); print('Config works')"
```

---

## GCP TPU Setup

### Create TPU VM
```bash
gcloud compute tpus tpu-vm create mft-experiment \
    --zone=us-central2-b \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base
```

### SSH into TPU
```bash
gcloud compute tpus tpu-vm ssh mft-experiment --zone=us-central2-b
```

### Setup Environment on TPU
```bash
# Clone repository
git clone https://github.com/chrisfrancisque/mft-tpu.git
cd mft-tpu

# Install PyTorch/XLA for TPU v4
pip install torch~=2.3.0 torch_xla[tpu]~=2.3.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Install dependencies
pip install -r requirements_tpu.txt

# Install OLMES for evaluation
git clone https://github.com/allenai/olmes.git
cd olmes && pip install -e . && cd ..

# Test TPU access
python3 -c "import torch_xla; import torch_xla.core.xla_model as xm; print(f'TPU devices: {xm.get_xla_supported_devices()}')"
```

### Create GCS Bucket
```bash
# Create bucket for outputs
gsutil mb -l us-central1 gs://chrisfrancisque-mft-outputs

# Update scripts with your bucket name
sed -i 's/YOUR_BUCKET/chrisfrancisque-mft-outputs/g' scripts/tpu_*.sh
```

### Pre-download Datasets
```bash
python3 << 'EOF'
from datasets import load_dataset

print("Caching datasets...")
datasets = [
    ("bigcode/evol-codealpaca-v1", None),
    ("sahil2801/CodeAlpaca-20k", None),
    ("nickrosh/Evol-Instruct-Code-80k-v1", None),
    ("openai_humaneval", None),
]

for name, config in datasets:
    print(f"Loading {name}...")
    ds = load_dataset(name, config, split="train" if "humaneval" not in name else "test")
    print(f"  Cached: {len(ds)} examples")

print("All datasets cached")
EOF
```

---

## Training

### Full Pipeline (Automated)
```bash
screen -S mft_pipeline
bash scripts/tpu_full_pipeline.sh chrisfrancisque-mft-outputs gemma_2b_coding

# Detach: Ctrl+A then D
# Reattach: screen -r mft_pipeline
```

### Step-by-Step Execution

#### 1. FFT Training
```bash
screen -S fft
bash scripts/tpu_train_fft.sh \
    configs/gemma_2b_coding_fft.yaml \
    gs://chrisfrancisque-mft-outputs/outputs/gemma_2b_coding_fft
```

#### 2. Evaluate FFT
```bash
bash scripts/tpu_evaluate.sh \
    gs://chrisfrancisque-mft-outputs/outputs/gemma_2b_coding_fft/checkpoint-best \
    gs://chrisfrancisque-mft-outputs/eval/fft
```

#### 3. MFT Training
```bash
screen -S mft
bash scripts/tpu_train_mft.sh \
    configs/gemma_2b_coding_mft.yaml \
    gs://chrisfrancisque-mft-outputs/outputs/gemma_2b_coding_fft/checkpoint-best \
    gs://chrisfrancisque-mft-outputs/outputs/gemma_2b_coding_mft
```

#### 4. Evaluate MFT
```bash
bash scripts/tpu_evaluate.sh \
    gs://chrisfrancisque-mft-outputs/outputs/gemma_2b_coding_mft/mask_applied \
    gs://chrisfrancisque-mft-outputs/eval/mft
```

#### 5. Evaluate Pretrained Baseline
```bash
bash scripts/tpu_evaluate.sh \
    google/gemma-2-2b \
    gs://chrisfrancisque-mft-outputs/eval/pretrained
```

---

## Monitoring

### Check Training Progress
```bash
# View logs
gsutil cat gs://chrisfrancisque-mft-outputs/outputs/gemma_2b_coding_fft/training.log | tail -50

# List checkpoints
gsutil ls gs://chrisfrancisque-mft-outputs/outputs/gemma_2b_coding_fft/checkpoint-*
```

### View Evaluation Results
```bash
gsutil cat gs://chrisfrancisque-mft-outputs/eval/fft/results.json
gsutil cat gs://chrisfrancisque-mft-outputs/eval/mft/results.json
gsutil cat gs://chrisfrancisque-mft-outputs/eval/pretrained/results.json
```

---

## Troubleshooting

### TPU Out of Memory
- Reduce `batch_size_per_core` from 4 to 2 in config files
- Ensure `gradient_checkpointing: true` is enabled in TPU config section

### Training Too Slow
- Check TPU utilization: `python3 -c "import torch_xla; torch_xla.core.xla_model.print_metrics()"`
- Verify datasets are pre-cached

### OLMES Evaluation Fails
- Verify model path is correct and checkpoint exists
- Ensure HumanEval dataset is cached locally
- Try running with `--batch_size 1`

---

## Cleanup

```bash
# Delete TPU VM
gcloud compute tpus tpu-vm delete mft-experiment --zone=us-central2-b

# Delete GCS bucket (optional)
gsutil -m rm -r gs://chrisfrancisque-mft-outputs
```
