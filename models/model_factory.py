"""
Model factory for loading and preparing models for FFT/MFT training.
Handles model initialization, masking setup, and state management.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel
)
from huggingface_hub import HfFolder
from typing import Optional, Dict, Any, Tuple
import logging
import os
from pathlib import Path

from .masked_layers import replace_linear_with_masked, MaskedLinear

logger = logging.getLogger(__name__)


class MFTModelWrapper(nn.Module):
    """Wrapper for models with MFT functionality."""
    
    def __init__(self, base_model: PreTrainedModel, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.mft_enabled = False
        
    def forward(self, **kwargs):
        """Forward pass through the base model."""
        return self.base_model(**kwargs)
    
    def enable_mft(self):
        """Enable MFT mode - freeze weights and enable mask learning."""
        self.mft_enabled = True
        
        # Enable masks in all MaskedLinear layers
        for module in self.base_model.modules():
            if isinstance(module, MaskedLinear):
                module.enable_mask()
        
        logger.info("MFT mode enabled - weights frozen, mask learning active")
    
    def disable_mft(self):
        """Disable MFT mode - return to normal training."""
        self.mft_enabled = False
        
        # Disable masks in all MaskedLinear layers
        for module in self.base_model.modules():
            if isinstance(module, MaskedLinear):
                module.disable_mask()
        
        logger.info("MFT mode disabled - normal training mode")
    
    def apply_masks_permanently(self):
        """Apply learned masks permanently to the model."""
        total_params = 0
        masked_params = 0
        
        for name, module in self.base_model.named_modules():
            if isinstance(module, MaskedLinear):
                stats = module.get_sparsity_stats()
                total_params += stats['total_params']
                masked_params += stats['total_params'] - stats['active_params']
                module.apply_mask_permanently()
                
        overall_sparsity = masked_params / total_params if total_params > 0 else 0
        logger.info(f"Applied masks permanently. Overall sparsity: {overall_sparsity:.4f}")
        
        return overall_sparsity
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Get count of trainable parameters."""
        trainable = 0
        total = 0
        
        for param in self.base_model.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        
        return {
            'trainable': trainable,
            'total': total,
            'percentage': (trainable / total * 100) if total > 0 else 0
        }
    
    def save_pretrained(self, save_path: str):
        """Save the model with proper handling of masked layers."""
        os.makedirs(save_path, exist_ok=True)

        # Move model to CPU before saving (required for XLA/TPU tensors)
        original_device = next(self.base_model.parameters()).device
        if str(original_device).startswith('xla'):
            import torch_xla.core.xla_model as xm
            xm.mark_step()  # Ensure all operations are complete
            self.base_model = self.base_model.cpu()

        # Save the base model
        self.base_model.save_pretrained(save_path)

        # Move model back to original device
        if str(original_device).startswith('xla'):
            self.base_model = self.base_model.to(original_device)
        
        # Save mask scores separately if in MFT mode
        if self.mft_enabled:
            mask_scores = {}
            for name, module in self.base_model.named_modules():
                if isinstance(module, MaskedLinear):
                    mask_scores[name] = {
                        'scores': module.scores.detach().cpu(),
                        'mask': module.mask.detach().cpu(),
                        'sparsity': module.sparsity
                    }
            
            if mask_scores:
                torch.save(mask_scores, os.path.join(save_path, 'mask_scores.pt'))
                logger.info(f"Saved mask scores to {save_path}/mask_scores.pt")
    
    def load_mask_scores(self, load_path: str):
        """Load previously learned mask scores."""
        mask_file = os.path.join(load_path, 'mask_scores.pt')
        if os.path.exists(mask_file):
            mask_scores = torch.load(mask_file, map_location='cpu')
            
            for name, module in self.base_model.named_modules():
                if isinstance(module, MaskedLinear) and name in mask_scores:
                    module.scores.data = mask_scores[name]['scores'].to(module.scores.device)
                    module.mask.data = mask_scores[name]['mask'].to(module.mask.device)
                    logger.info(f"Loaded mask scores for {name}")
            
            logger.info("Successfully loaded all mask scores")
        else:
            logger.warning(f"No mask scores found at {mask_file}")


class ModelFactory:
    """Factory for creating and loading models for FFT/MFT training."""
    
    @staticmethod
    def load_base_model(config) -> Tuple[PreTrainedModel, AutoTokenizer]:
        """Load a pre-trained model for FFT or as base for MFT."""

        logger.info(f"Loading base model: {config.model.model_name}")

        # Get HuggingFace token for gated models
        hf_token = HfFolder.get_token()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.data.tokenizer_name or config.model.model_name,
            revision=config.model.model_revision,
            use_fast=config.data.use_fast_tokenizer,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
            token=hf_token  # Use saved HF token for gated models like LLaMA
        )
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        # Add chat template if not present
        if tokenizer.chat_template is None:
            # LLaMA2 uses [INST] format
            if "llama" in config.model.model_name.lower():
                tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}"
                logger.info("Set default LLaMA2 chat template")
            else:
                # Gemma and similar models
                tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}{% elif message['role'] == 'assistant' %}{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>\n' }}{% endif %}{% endfor %}"
                logger.info("Set default Gemma chat template")

        # Load model config
        model_config = AutoConfig.from_pretrained(
            config.model.model_name,
            revision=config.model.model_revision,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
            token=hf_token  # Use saved HF token for gated models like LLaMA
        )
        
        # Update model config with training settings
        model_config.use_cache = False  # Disable KV cache for training
        model_config.gradient_checkpointing = config.tpu.gradient_checkpointing
        
        # Load model
        # Handle different precision formats
        dtype = torch.float32
        if config.tpu.mixed_precision in ["bfloat16", "bf16"]:
            dtype = torch.bfloat16
        elif config.tpu.mixed_precision == "float16":
            dtype = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            config=model_config,
            revision=config.model.model_revision,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
            torch_dtype=dtype,
            token=hf_token  # Use saved HF token for gated models like LLaMA
        )
        
        # Resize token embeddings if needed
        model.resize_token_embeddings(len(tokenizer))
        
        logger.info(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer
    
    @staticmethod
    def prepare_model_for_fft(config) -> Tuple[MFTModelWrapper, AutoTokenizer]:
        """Prepare a model for full fine-tuning."""
        
        # Load base model
        model, tokenizer = ModelFactory.load_base_model(config)
        
        # Wrap in MFTModelWrapper (even for FFT, for consistency)
        wrapped_model = MFTModelWrapper(model, config)
        
        # For FFT, all parameters should be trainable
        for param in wrapped_model.parameters():
            param.requires_grad = True
        
        param_stats = wrapped_model.get_trainable_params()
        logger.info(
            f"Prepared model for FFT. "
            f"Trainable: {param_stats['trainable']:,} / {param_stats['total']:,} "
            f"({param_stats['percentage']:.2f}%)"
        )
        
        return wrapped_model, tokenizer
    
    @staticmethod
    def prepare_model_for_mft(config, base_model_path: str) -> Tuple[MFTModelWrapper, AutoTokenizer]:
        """Prepare a model for mask fine-tuning."""
        
        logger.info(f"Loading FFT model from: {base_model_path}")
        
        # Load the FFT model
        model_config = AutoConfig.from_pretrained(
            base_model_path,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            config=model_config,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir,
            torch_dtype=torch.bfloat16 if config.tpu.mixed_precision == "bfloat16" else torch.float32
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=config.data.use_fast_tokenizer,
            trust_remote_code=config.model.trust_remote_code,
            cache_dir=config.model.cache_dir
        )
        
        # Replace Linear layers with MaskedLinear in target layers
        logger.info(f"Replacing Linear layers in layers {config.mask.masked_layers} with MaskedLinear")
        model = replace_linear_with_masked(model, config)
        
        # Wrap model
        wrapped_model = MFTModelWrapper(model, config)
        
        # Enable MFT mode
        wrapped_model.enable_mft()
        
        param_stats = wrapped_model.get_trainable_params()
        logger.info(
            f"Prepared model for MFT. "
            f"Trainable (mask scores): {param_stats['trainable']:,} / {param_stats['total']:,} "
            f"({param_stats['percentage']:.2f}%)"
        )
        
        return wrapped_model, tokenizer
    
    @staticmethod
    def create_model(config) -> Tuple[MFTModelWrapper, AutoTokenizer]:
        """Create a model based on the experiment configuration."""
        
        if config.experiment_type == "fft":
            return ModelFactory.prepare_model_for_fft(config)
        elif config.experiment_type == "mft":
            if not config.base_model_path:
                raise ValueError("MFT requires base_model_path to be specified")
            return ModelFactory.prepare_model_for_mft(config, config.base_model_path)
        else:
            raise ValueError(f"Unknown experiment type: {config.experiment_type}")
    
    @staticmethod
    def load_checkpoint(
        model: MFTModelWrapper,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """Load a training checkpoint."""
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load mask scores if present
        if 'mask_scores' in checkpoint:
            model.load_mask_scores(checkpoint_path)
        
        logger.info("Checkpoint loaded successfully")
        
        return checkpoint