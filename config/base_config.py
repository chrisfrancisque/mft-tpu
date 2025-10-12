"""
Configuration system for MFT-TPU training.
Handles model, training, and TPU-specific configurations.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path

@dataclass
class TPUConfig:
    """TPU configuration"""
    use_tpu: bool = True
    tpu_name: Optional[str] = None
    tpu_zone: Optional[str] = None
    num_tpu_cores: int = 8
    mixed_precision: str = 'bfloat16'

    # TPU optimization settings
    batch_size_per_core: int = 16
    gradient_accumulation_steps: int = 4

    # XLA compilation settings
    xla_use_bf16: bool = True
    xla_tp_degree: int = 1

    # Memory Optimization
    gradient_checkpointing: bool = False
    optimizer_memory_efficient: bool = True


@dataclass
class ModelConfig:
    """Model Configuration"""
    model_name: str = "google/gemma-2b"
    model_revision: str = "main"

    #Model loading settings
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    cache_dir: str = "./model_cache"

    #Model architecture settings
    max_seq_length: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1

    # Mask setting
    vocab_size: Optional[int] = None


@dataclass
class MaskConfig:
    """Mask Fine-tuning configuration"""
    mask_type: str = 'local' #either can be local or global
    sparsity_ratio: float = 0.1 # 10% sparsity per paper

    # Layer specific masking
    masked_layers: List[int] = field(default_factory= lambda: [4,5,6,7])
    apply_to_attention: bool = True
    apply_to_mlp: bool = True

    #Mask learning settings
    mask_learning_rate: float = 1e-3
    straight_through_estimator: bool = True
    temperature: float = 1.0 #for soft masking

    # Learnable Score Initialization
    score_init_method: str = "kaiming" # kaiming, xavier, normal
    score_init_std: float = 0.01

@dataclass
class TrainingConfig:
        """Training Configuration"""

        #General Setting
        output_dir: str = "./outputs"
        experiment_name: str = "mft_experiments"
        seed: int = 42

        # Training parameters
        num_train_epochs: int = 2
        learning_rate: float = 2e-5
        warmup_ratio: float = 0.03
        weight_decay: float = 0.0
        max_grad_norm: float = 1.0

        # Optimizer settings
        optimizer: str = "adamw"
        adam_beta1: float = 0.9
        adam_beta2: float = 0.999
        adam_epsilon: float = 1e-8

         # Scheduler
        lr_scheduler_type: str = "linear"

        # Logging and saving
        logging_steps: int = 10
        save_steps: int = 500
        eval_steps: int = 500
        save_total_limit: int = 3

        # Evaluation
        evaluation_strategy: str = "steps"
        metric_for_best_model: str = "eval_loss"
        greater_is_better: bool = False

        # Early stopping
        early_stopping_patience: Optional[int] = None
        early_stopping_threshold: float = 0.0

@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset settings
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_mixer: Optional[Dict[str, float]] = None

    # Data paths
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    cache_dir: str = "./data_cache"

    # Processing settings
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessing_num_workers: int = 4

    # Tokenization
    tokenizer_name: Optional[str] = None
    use_fast_tokenizer: bool = True
    padding: str = "max_length"
    truncation: bool = True

    # Data loading
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tpu: TPUConfig = field(default_factory=TPUConfig)

    # Experiment metadata
    experiment_type: str = "mft"  # "fft" or "mft"
    domain: str = "math"  # "math", "coding", or "instruction"
    base_model_path: Optional[str] = None  # Path to FFT model for MFT

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create nested configs
        config = cls() 

        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'mask' in config_dict:
            config.mask = MaskConfig(**config_dict['mask'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'tpu' in config_dict:
            config.tpu = TPUConfig(**config_dict['tpu']) 


        # Set top-level attributes
        for key in ['experiment_type', 'domain', 'base_model_path']:
            if key in config_dict:
                setattr(config, key, config_dict[key]) 
        
        return config
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'experiment_type': self.experiment_type,
            'domain': self.domain,
            'base_model_path': self.base_model_path,
            'model': asdict(self.model),
            'mask': asdict(self.mask),
            'training': asdict(self.training),
            'data': asdict(self.data),
            'tpu': asdict(self.tpu)
        }

        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def get_output_dir(self) -> str:
        """Generate output directory path"""
        return os.path.join(
            self.training.output_dir,
            self.experiment_type,
            self.domain,
            self.training.experiment_name
        )
    
    def validate(self):
        """Validate configuration consistency"""
        # MFT requires base model
        if self.experiment_type == "mft" and not self.base_model_path:
            raise ValueError("MFT experiment requires base_model_path to be set")
        
        # Check domain-specific settings
        if self.domain == "math":
            # Math typically uses layers 4-7 for LLaMA2-7B equivalent
            pass
        elif self.domain == "coding":
            # Coding typically uses layers 20-23
            if self.mask.masked_layers == [4, 5, 6, 7]:
                self.mask.masked_layers = [20, 21, 22, 23]
        elif self.domain == "instruction":
            # Instruction typically uses layers 0-3
            if self.mask.masked_layers == [4, 5, 6, 7]:
                self.mask.masked_layers = [0, 1, 2, 3]
        
        # TPU batch size should be divisible by num cores
        total_batch = self.tpu.batch_size_per_core * self.tpu.num_tpu_cores
        if total_batch % self.tpu.num_tpu_cores != 0:
            raise ValueError(f"Total batch size {total_batch} not divisible by {self.tpu.num_tpu_cores} cores")
        
        return True
    



