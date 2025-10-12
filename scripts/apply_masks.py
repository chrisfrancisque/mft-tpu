"""
Script to permanently apply learned masks to a model after MFT training.
This creates the final optimized model with harmful weights removed.
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import torch
import json
from pathlib import Path
from typing import Dict

from models.model_factory import ModelFactory
from models.masked_layers import MaskedLinear
from transformers import AutoTokenizer
from config.base_config import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_mask_distribution(model) -> Dict:
    """Analyze the distribution of learned masks."""
    stats = {
        'layer_stats': {},
        'total_params': 0,
        'total_masked': 0,
        'total_active': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, MaskedLinear):
            layer_stats = module.get_sparsity_stats()
            stats['layer_stats'][name] = {
                'total': layer_stats['total_params'],
                'active': layer_stats['active_params'],
                'masked': layer_stats['total_params'] - layer_stats['active_params'],
                'sparsity': layer_stats['sparsity']
            }
            stats['total_params'] += layer_stats['total_params']
            stats['total_active'] += layer_stats['active_params']
            stats['total_masked'] += layer_stats['total_params'] - layer_stats['active_params']
    
    if stats['total_params'] > 0:
        stats['overall_sparsity'] = stats['total_masked'] / stats['total_params']
    else:
        stats['overall_sparsity'] = 0.0
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Apply learned masks permanently to create final model")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to MFT checkpoint with learned masks')
    parser.add_argument('--output_path', type=str,
                       help='Output path for final model (default: model_path/final_masked)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze masks without applying them')
    parser.add_argument('--save_analysis', type=str,
                       help='Path to save mask analysis JSON')
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Set output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = model_path / "final_masked"
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load config if available
    config_path = model_path / "config.json"
    if config_path.exists():
        # Load model with transformers
        from transformers import AutoModelForCausalLM, AutoConfig
        
        model_config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=model_config,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # Try to load with custom config
        training_state_path = model_path / "training_state.pt"
        if training_state_path.exists():
            checkpoint = torch.load(training_state_path, map_location='cpu')
            config_dict = checkpoint.get('config', {})
            
            # Reconstruct config
            from config.base_config import ExperimentConfig
            config = ExperimentConfig()
            # Update with saved config (this is simplified - you'd need proper reconstruction)
            
            model, tokenizer = ModelFactory.prepare_model_for_mft(
                config, str(model_path.parent)
            )
            
            # Load mask scores
            mask_scores_path = model_path / "mask_scores.pt"
            if mask_scores_path.exists():
                model.load_mask_scores(str(model_path))
        else:
            raise ValueError("Cannot determine model format. Missing config.json or training_state.pt")
    
    # Check if model has masked layers
    has_masked_layers = any(isinstance(m, MaskedLinear) for m in model.modules())
    if not has_masked_layers:
        logger.error("Model does not contain any MaskedLinear layers!")
        return
    
    # Analyze mask distribution
    logger.info("Analyzing learned masks...")
    mask_stats = analyze_mask_distribution(model)
    
    logger.info(f"Overall sparsity: {mask_stats['overall_sparsity']:.4f}")
    logger.info(f"Total parameters: {mask_stats['total_params']:,}")
    logger.info(f"Masked parameters: {mask_stats['total_masked']:,}")
    logger.info(f"Active parameters: {mask_stats['total_active']:,}")
    
    # Show per-layer statistics
    logger.info("\nPer-layer mask statistics:")
    for layer_name, layer_stats in mask_stats['layer_stats'].items():
        logger.info(f"  {layer_name}:")
        logger.info(f"    Sparsity: {layer_stats['sparsity']:.4f}")
        logger.info(f"    Masked: {layer_stats['masked']:,} / {layer_stats['total']:,}")
    
    # Save analysis if requested
    if args.save_analysis:
        with open(args.save_analysis, 'w') as f:
            json.dump(mask_stats, f, indent=2)
        logger.info(f"Saved mask analysis to: {args.save_analysis}")
    
    # Apply masks if not analyze-only
    if not args.analyze_only:
        logger.info("Applying masks permanently...")
        
        # Apply masks to all MaskedLinear layers
        for name, module in model.named_modules():
            if isinstance(module, MaskedLinear):
                module.apply_mask_permanently()
                logger.info(f"Applied mask to: {name}")
        
        # Save the final model
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving final masked model to: {output_path}")
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        # Save mask statistics with the model
        stats_path = output_path / "mask_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(mask_stats, f, indent=2)
        
        logger.info("Successfully created final masked model!")
        logger.info(f"Model saved to: {output_path}")
        logger.info(f"Final sparsity: {mask_stats['overall_sparsity']:.4f}")
    else:
        logger.info("Analysis complete (masks not applied)")


if __name__ == "__main__":
    main()