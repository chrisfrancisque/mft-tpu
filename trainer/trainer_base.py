# trainer/trainer_base.py
"""
Base trainer implementation for FFT and MFT training on TPU/CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any, Tuple
import logging
import os
from pathlib import Path
from tqdm import tqdm
import time
import json

import sys
sys.path.append('..')
from utils.device_utils import get_device_manager
from models.masked_layers import MaskedLinear

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer for both FFT and MFT training."""
    
    def __init__(
        self,
        model,
        tokenizer,
        config,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Get device manager
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        # Create output directory
        self.output_dir = Path(self.config.get_output_dir())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _create_optimizer(self):
        """Create optimizer for training."""
        # Get parameters to optimize
        if self.config.experiment_type == "mft":
            # For MFT, only optimize mask scores
            params_to_optimize = []
            for name, param in self.model.named_parameters():
                if 'scores' in name and param.requires_grad:
                    params_to_optimize.append(param)
                    logger.info(f"Optimizing parameter: {name}")
        else:
            # For FFT, optimize all parameters
            params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        
        if not params_to_optimize:
            raise ValueError("No parameters to optimize!")
        
        # Create optimizer
        if self.config.training.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.config.training.learning_rate,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                eps=self.config.training.adam_epsilon,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = len(self.train_dataloader) * self.config.training.num_train_epochs
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        
        if self.config.training.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=num_warmup_steps
            )
        elif self.config.training.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _setup_logging(self):
        """Setup logging and metrics tracking."""
        self.log_file = self.output_dir / "training_log.jsonl"
        self.metrics_history = []
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute training loss."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Compute additional metrics
        metrics = {
            'loss': loss.item(),
            'perplexity': torch.exp(loss).item()
        }
        
        # Add MFT-specific metrics if applicable
        if self.config.experiment_type == "mft":
            sparsity_stats = self._get_sparsity_stats()
            metrics.update(sparsity_stats)
        
        return loss, metrics
    
    def _get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics for MFT models."""
        stats = {
            'total_params': 0,
            'masked_params': 0,
            'sparsity': 0.0
        }
        
        for module in self.model.modules():
            if isinstance(module, MaskedLinear):
                module_stats = module.get_sparsity_stats()
                stats['total_params'] += module_stats['total_params']
                stats['masked_params'] += module_stats['total_params'] - module_stats['active_params']
        
        if stats['total_params'] > 0:
            stats['sparsity'] = stats['masked_params'] / stats['total_params']
        
        return stats
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'perplexity': 0.0,
            'samples': 0
        }
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.device_manager.is_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            # Compute loss
            loss, metrics = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            batch_size = batch['input_ids'].size(0)
            epoch_metrics['loss'] += loss.item() * batch_size
            epoch_metrics['perplexity'] += metrics['perplexity'] * batch_size
            epoch_metrics['samples'] += batch_size
            
            # Update progress bar
            if step % self.config.training.logging_steps == 0:
                avg_loss = epoch_metrics['loss'] / epoch_metrics['samples']
                progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})
            
            # Synchronize for TPU
            if self.device_manager.is_tpu:
                import torch_xla.core.xla_model as xm
                xm.mark_step()
            
            self.global_step += 1
            
            # Save checkpoint
            if self.global_step % self.config.training.save_steps == 0:
                self.save_checkpoint()
        
        # Compute epoch averages
        for key in ['loss', 'perplexity']:
            epoch_metrics[key] /= epoch_metrics['samples']
        
        return epoch_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        eval_metrics = {
            'eval_loss': 0.0,
            'eval_perplexity': 0.0,
            'samples': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                disable=not self.device_manager.is_main_process
            ):
                loss, metrics = self.compute_loss(batch)
                
                batch_size = batch['input_ids'].size(0)
                eval_metrics['eval_loss'] += loss.item() * batch_size
                eval_metrics['eval_perplexity'] += metrics['perplexity'] * batch_size
                eval_metrics['samples'] += batch_size
        
        # Compute averages
        for key in ['eval_loss', 'eval_perplexity']:
            eval_metrics[key] /= eval_metrics['samples']
        
        return eval_metrics
    
    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.training.num_train_epochs} epochs")
        logger.info(f"Total training steps: {len(self.train_dataloader) * self.config.training.num_train_epochs}")
        
        for epoch in range(self.config.training.num_train_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            if self.eval_dataloader and (epoch + 1) % self.config.training.eval_steps == 0:
                eval_metrics = self.evaluate()
                train_metrics.update(eval_metrics)
                
                # Check for best model
                if eval_metrics['eval_loss'] < self.best_metric:
                    self.best_metric = eval_metrics['eval_loss']
                    self.save_checkpoint(is_best=True)
            
            # Log metrics
            self.log_metrics(train_metrics, epoch)
            
            # Print metrics
            if self.device_manager.is_main_process:
                logger.info(f"Epoch {epoch} metrics: {train_metrics}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        if not self.device_manager.is_main_process:
            return
        
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        if is_best:
            checkpoint_dir = self.output_dir / "best_model"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(str(checkpoint_dir))
        self.tokenizer.save_pretrained(str(checkpoint_dir))
        
        # Save optimizer and scheduler states
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.__dict__
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / 'training_state.pt')
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to file."""
        if not self.device_manager.is_main_process:
            return
        
        log_entry = {
            'epoch': epoch,
            'global_step': self.global_step,
            'timestamp': time.time(),
            **metrics
        }
        
        self.metrics_history.append(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')