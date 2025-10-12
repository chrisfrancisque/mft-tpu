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


def main():
    parser = argparse.ArgumentParser(description="Full Fine-Tuning Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--resume_from', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Setup TPU environment if available
    is_tpu = setup_tpu_env.setup_tpu_environment()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    config.experiment_type = "fft"  # Ensure FFT mode
    
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    # Validate config
    config.validate()
    
    logger.info(f"Starting FFT training for domain: {config.domain}")
    logger.info(f"Output directory: {config.get_output_dir()}")
    
    # Initialize device
    device_manager = get_device_manager()
    logger.info(f"Using device: {device_manager.device_config.device_type}")
    
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
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        ModelFactory.load_checkpoint(
            model=model,
            checkpoint_path=args.resume_from,
            optimizer=trainer.optimizer
        )
    
    # Start training
    trainer.train()
    
    # Finalize and save
    trainer.finalize_training()
    
    logger.info("FFT training completed successfully!")


if __name__ == "__main__":
    main()