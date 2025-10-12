"""
FFT (Full Fine-Tuning) trainer implementation.
"""

import torch
import logging
from typing import Dict, Any, Tuple
from .trainer_base import BaseTrainer

logger = logging.getLogger(__name__)


class FFTTrainer(BaseTrainer):
    """Trainer for Full Fine-Tuning."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Ensure model is in normal training mode (not MFT)
        if hasattr(self.model, 'disable_mft'):
            self.model.disable_mft()
            logger.info("FFT mode enabled - all parameters trainable")
        
        # Log trainable parameters
        self._log_parameter_stats()
    
    def _log_parameter_stats(self):
        """Log statistics about trainable parameters."""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute standard language modeling loss."""
        loss, metrics = super().compute_loss(batch)
        
        # Add any FFT-specific metrics
        metrics['grad_norm'] = self._compute_grad_norm()
        
        return loss, metrics
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm for monitoring."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def finalize_training(self):
        """Finalize FFT training."""
        logger.info("Finalizing FFT training")
        
        # Save final model
        final_dir = self.output_dir / "final_fft_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(str(final_dir))
        else:
            # For wrapped models
            if hasattr(self.model, 'base_model'):
                self.model.base_model.save_pretrained(str(final_dir))
            else:
                torch.save(self.model.state_dict(), final_dir / 'pytorch_model.bin')
        
        self.tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Saved final FFT model to {final_dir}")