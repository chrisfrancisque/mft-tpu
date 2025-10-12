# trainer/mft_trainer.py
"""
MFT-specific trainer with mask learning capabilities.
"""

import torch
from typing import Dict, Any, Tuple
import logging
from .trainer_base import BaseTrainer
from models.masked_layers import MaskedLinear

logger = logging.getLogger(__name__)


class MFTTrainer(BaseTrainer):
    """Trainer specifically for Mask Fine-Tuning."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Ensure model is in MFT mode
        if hasattr(self.model, 'enable_mft'):
            self.model.enable_mft()
            logger.info("MFT mode enabled for training")
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss with MFT-specific regularization."""
        loss, metrics = super().compute_loss(batch)
        
        # Add L0 regularization for sparsity (optional)
        if hasattr(self.config.mask, 'l0_lambda') and self.config.mask.l0_lambda > 0:
            l0_loss = self._compute_l0_regularization()
            loss = loss + self.config.mask.l0_lambda * l0_loss
            metrics['l0_loss'] = l0_loss.item()
        
        return loss, metrics
    
    def _compute_l0_regularization(self) -> torch.Tensor:
        """Compute L0 regularization to encourage sparsity."""
        l0_loss = 0.0
        
        for module in self.model.modules():
            if isinstance(module, MaskedLinear) and module.use_mask:
                # Use scores as proxy for L0
                scores_normalized = torch.sigmoid(module.scores / self.config.mask.temperature)
                l0_loss += scores_normalized.sum()
        
        return l0_loss
    
    def finalize_training(self):
        """Apply learned masks permanently after training."""
        logger.info("Finalizing MFT training - applying masks permanently")
        
        if hasattr(self.model, 'apply_masks_permanently'):
            sparsity = self.model.apply_masks_permanently()
            logger.info(f"Final model sparsity: {sparsity:.4f}")
            
            # Save final model
            final_dir = self.output_dir / "final_masked_model"
            final_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(final_dir))
            self.tokenizer.save_pretrained(str(final_dir))
            logger.info(f"Saved final masked model to {final_dir}")