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

        # Properly unwrap and save FSDP model
        model_to_save = self._unwrap_model_for_saving()

        if model_to_save is not None:
            model_to_save.save_pretrained(str(final_dir))
            self.tokenizer.save_pretrained(str(final_dir))
            logger.info(f"Saved final FFT model to {final_dir}")
        else:
            logger.warning("Skipping save on non-master rank")

    def _unwrap_model_for_saving(self):
        """Unwrap model from FSDP and MFT wrappers for saving."""
        model = self.model

        # Unwrap from MFTModelWrapper if present
        if hasattr(model, 'base_model'):
            logger.info("Unwrapping from MFTModelWrapper...")
            model = model.base_model

        # Handle FSDP unwrapping on TPU
        if self.device_manager.is_tpu:
            try:
                from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, consolidate_sharded_model_checkpoints
                import torch_xla.core.xla_model as xm

                if isinstance(model, FSDP):
                    logger.info("Detected FSDP model, consolidating shards...")

                    # Use FSDP's built-in consolidation if available
                    try:
                        # Get consolidated state dict on rank 0
                        consolidated_state_dict = consolidate_sharded_model_checkpoints(
                            ckpt_prefix=str(self.output_dir / "temp_shard"),
                            ckpt_suffix="_rank-*-of-*.pth",
                        )

                        if xm.is_master_ordinal() and consolidated_state_dict is not None:
                            # Get underlying module
                            unwrapped_model = model.module
                            unwrapped_model.load_state_dict(consolidated_state_dict)
                            logger.info("Loaded consolidated state dict")
                            return unwrapped_model
                        else:
                            return None

                    except (ImportError, AttributeError, TypeError):
                        # Fallback: manually consolidate by moving to CPU
                        logger.info("Using manual consolidation (moving to CPU)...")

                        if not xm.is_master_ordinal():
                            return None

                        # Get underlying module
                        unwrapped_model = model.module

                        # Force all parameters to CPU
                        unwrapped_model = unwrapped_model.cpu()

                        logger.info("Moved model to CPU for saving")
                        return unwrapped_model

            except (ImportError, AttributeError) as e:
                logger.warning(f"FSDP handling failed: {e}, using model as-is")

        return model