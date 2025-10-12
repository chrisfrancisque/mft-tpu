"""
Main script for Mask Fine-Tuning (MFT) training.
Loads a fully fine-tuned model and learns binary masks to identify harmful parameters.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
import torch
import json

from config.base_config import ExperimentConfig
from models.model_factory import ModelFactory
from data.data_mixers import DataMixer
from trainer.mft_trainer import MFTTrainer
from utils.device_utils import get_device_manager
import setup_tpu_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_base_model(base_model_path: str) -> dict:
    """Validate that the base model exists and get its info."""
    base_path = Path(base_model_path)
    
    if not base_path.exists():
        raise ValueError(f"Base model path does not exist: {base_model_path}")
    
    # Check for required files
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json']
    missing_files = []
    
    for file in required_files:
        if not (base_path / file).exists():
            # Check for safetensors alternative
            if file == 'pytorch_model.bin' and (base_path / 'model.safetensors').exists():
                continue
            missing_files.append(file)
    
    if missing_files:
        logger.warning(f"Missing files in base model: {missing_files}")
    
    # Try to load training info if available
    training_info = {}
    training_state_path = base_path / 'training_state.pt'
    if training_state_path.exists():
        try:
            state = torch.load(training_state_path, map_location='cpu')
            training_info = {
                'final_epoch': state.get('epoch', 'unknown'),
                'final_step': state.get('global_step', 'unknown'),
                'best_metric': state.get('best_metric', 'unknown')
            }
            logger.info(f"Base model training info: {training_info}")
        except Exception as e:
            logger.warning(f"Could not load training state: {e}")
    
    return training_info


def main():
    parser = argparse.ArgumentParser(description="Mask Fine-Tuning (MFT) Training")
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    parser.add_argument('--base_model_path', type=str, required=True,
                       help='Path to fully fine-tuned model to use as base')
    parser.add_argument('--output_dir', type=str, 
                       help='Override output directory')
    parser.add_argument('--resume_from', type=str, 
                       help='Resume MFT training from checkpoint')
    parser.add_argument('--mask_layers', type=str,
                       help='Comma-separated list of layer indices to mask (e.g., "4,5,6,7")')
    parser.add_argument('--sparsity_ratio', type=float, default=0.1,
                       help='Target sparsity ratio (default: 0.1)')
    parser.add_argument('--apply_masks_after_training', action='store_true',
                       help='Automatically apply masks permanently after training')
    args = parser.parse_args()
    
    # Setup TPU environment if available
    is_tpu = setup_tpu_env.setup_tpu_environment()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    config.experiment_type = "mft"  # Ensure MFT mode
    config.base_model_path = args.base_model_path
    
    # Override config with command line arguments
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    if args.mask_layers:
        config.mask.masked_layers = [int(x) for x in args.mask_layers.split(',')]
        logger.info(f"Overriding masked layers to: {config.mask.masked_layers}")
    
    config.mask.sparsity_ratio = args.sparsity_ratio
    
    # Validate base model
    logger.info(f"Validating base FFT model: {config.base_model_path}")
    base_model_info = validate_base_model(config.base_model_path)
    
    # Validate config
    config.validate()
    
    # Update output directory to include MFT info
    mft_suffix = f"mft_layers_{'_'.join(map(str, config.mask.masked_layers))}_sp{int(config.mask.sparsity_ratio*100)}"
    config.training.experiment_name = f"{config.training.experiment_name}_{mft_suffix}"
    
    logger.info(f"Starting MFT training for domain: {config.domain}")
    logger.info(f"Base model: {config.base_model_path}")
    logger.info(f"Target layers: {config.mask.masked_layers}")
    logger.info(f"Target sparsity: {config.mask.sparsity_ratio}")
    logger.info(f"Output directory: {config.get_output_dir()}")
    
    # Initialize device
    device_manager = get_device_manager()
    logger.info(f"Using device: {device_manager.device_config.device_type}")
    
    # Create model and tokenizer for MFT
    logger.info("Loading base model and preparing for MFT...")
    model, tokenizer = ModelFactory.create_model(config)
    
    # Log parameter statistics
    param_stats = model.get_trainable_params()
    logger.info(f"Model parameters - Total: {param_stats['total']:,}, "
                f"Trainable (mask scores): {param_stats['trainable']:,} "
                f"({param_stats['percentage']:.2f}%)")
    
    # Create data loaders
    logger.info(f"Loading datasets for domain: {config.domain}")
    dataloaders = DataMixer.create_domain_dataloaders(
        config=config,
        tokenizer=tokenizer,
        domain=config.domain
    )
    
    # Initialize MFT trainer
    trainer = MFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders.get('validation')
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming MFT training from checkpoint: {args.resume_from}")
        checkpoint = ModelFactory.load_checkpoint(
            model=model,
            checkpoint_path=args.resume_from,
            optimizer=trainer.optimizer
        )
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.current_epoch = checkpoint.get('epoch', 0)
        trainer.best_metric = checkpoint.get('best_metric', float('inf'))
    
    # Start training
    logger.info("Starting MFT training...")
    trainer.train()
    
    # Get final sparsity statistics
    final_stats = trainer._get_sparsity_stats()
    logger.info(f"Final sparsity achieved: {final_stats['sparsity']:.4f}")
    logger.info(f"Total masked parameters: {final_stats['masked_params']:,}")
    
    # Apply masks permanently if requested
    if args.apply_masks_after_training:
        logger.info("Applying learned masks permanently...")
        trainer.finalize_training()
        
        # Save mask statistics
        mask_stats_path = Path(config.get_output_dir()) / "mask_statistics.json"
        with open(mask_stats_path, 'w') as f:
            json.dump({
                'base_model': config.base_model_path,
                'domain': config.domain,
                'masked_layers': config.mask.masked_layers,
                'target_sparsity': config.mask.sparsity_ratio,
                'achieved_sparsity': final_stats['sparsity'],
                'total_params': final_stats['total_params'],
                'masked_params': final_stats['masked_params'],
                'base_model_info': base_model_info
            }, f, indent=2)
        logger.info(f"Saved mask statistics to {mask_stats_path}")
    else:
        # Just save the checkpoint with learned masks
        trainer.save_checkpoint(is_best=True)
        logger.info("Saved final MFT checkpoint (masks not permanently applied)")
        logger.info("To apply masks, run: python scripts/apply_masks.py --model_path <checkpoint_path>")
    
    logger.info("MFT training completed successfully!")


if __name__ == "__main__":
    main()