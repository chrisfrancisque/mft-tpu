"""
Main script for Full Fine-Tuning (FFT) training.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

from config.base_config import ExperimentConfig
from models.model_factory import ModelFactory
from data.data_mixers import DataMixer
from trainer.fft_trainer import FFTTrainer
from utils.device_utils import get_device_manager
import setup_tpu_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global args for TPU multiprocessing
_global_args = None


def train_function(index=None):
    """Training function that can be spawned on multiple TPU cores."""
    global _global_args
    args = _global_args

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    config.experiment_type = "fft"  # Ensure FFT mode

    if args.output_dir:
        config.training.output_dir = args.output_dir

    # Validate config
    config.validate()

    # Initialize device
    device_manager = get_device_manager()

    # Log only on main process
    if device_manager.is_main_process:
        logger.info(f"Starting FFT training for domain: {config.domain}")
        logger.info(f"Output directory: {config.get_output_dir()}")
        logger.info(f"Using device: {device_manager.device_config.device_type}")
        logger.info(f"World size: {device_manager.device_config.world_size}")

    # Create model and tokenizer
    model, tokenizer = ModelFactory.create_model(config)

    # Create data loaders
    dataloaders = DataMixer.create_domain_dataloaders(
        config=config,
        tokenizer=tokenizer,
        domain=config.domain
    )

    # Initialize trainer
    trainer = FFTTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataloader=dataloaders['train'],
        eval_dataloader=dataloaders.get('validation')
    )

    # Resume from checkpoint if specified
    if args.resume_from:
        if device_manager.is_main_process:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ModelFactory.load_checkpoint(
            model=model,
            checkpoint_path=args.resume_from,
            optimizer=trainer.optimizer
        )

    # Start training
    trainer.train()

    # Finalize and save (only on main process)
    if device_manager.is_main_process:
        trainer.finalize_training()
        logger.info("FFT training completed successfully!")


def main():
    global _global_args

    parser = argparse.ArgumentParser(description="Full Fine-Tuning Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--resume_from', type=str, help='Resume from checkpoint')
    args = parser.parse_args()

    _global_args = args

    # Setup TPU environment if available
    is_tpu = setup_tpu_env.setup_tpu_environment()

    # Check if we should use TPU multiprocessing
    if is_tpu:
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp
            # Launch training on all TPU cores
            logger.info("Launching multi-core TPU training...")
            xmp.spawn(train_function, args=())
        except Exception as e:
            logger.error(f"Failed to spawn TPU processes: {e}")
            logger.info("Falling back to single-process mode...")
            train_function()
    else:
        # Run single process for CPU/GPU
        train_function()


if __name__ == "__main__":
    main()