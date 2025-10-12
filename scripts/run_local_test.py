# scripts/run_local_test.py
"""
Quick local test to verify the setup works.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports work."""
    try:
        from config.base_config import ExperimentConfig
        from models.model_factory import ModelFactory
        from models.masked_layers import MaskedLinear
        from data.dataset_loaders import DomainDatasetLoader
        from data.data_mixers import DataMixer
        from trainer.trainer_base import BaseTrainer
        from trainer.fft_trainer import FFTTrainer
        from trainer.mft_trainer import MFTTrainer
        from utils.device_utils import get_device_manager
        from evaluate.evaluator import Evaluator
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from config.base_config import ExperimentConfig
        
        # Test creating FFT config (doesn't need base_model_path)
        config = ExperimentConfig()
        config.experiment_type = "fft"  # Set to FFT to avoid MFT validation error
        config.validate()
        logger.info("✓ FFT config validation works")
        
        # Test MFT config with proper base_model_path
        mft_config = ExperimentConfig()
        mft_config.experiment_type = "mft"
        mft_config.base_model_path = "/dummy/path/for/testing"
        mft_config.validate()
        logger.info("✓ MFT config validation works")
        
        # Test loading from YAML if exists
        config_path = Path("configs/gemma_2b_math_fft.yaml")
        if config_path.exists():
            config = ExperimentConfig.from_yaml(str(config_path))
            config.validate()
            logger.info(f"✓ Config loaded from YAML: {config.training.experiment_name}")
        else:
            logger.info("✓ Config system works (no YAML file to test)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False

def test_device():
    """Test device detection."""
    try:
        from utils.device_utils import get_device_manager
        
        device_manager = get_device_manager(force_cpu=True)  # Force CPU for testing
        logger.info(f"✓ Device: {device_manager.device_config.device_type}")
        
        # Test CUDA detection if available
        if torch.cuda.is_available():
            logger.info(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            logger.info("  MPS (Apple Silicon) available")
        else:
            logger.info("  Running on CPU")
        
        return True
    except Exception as e:
        logger.error(f"✗ Device test failed: {e}")
        return False

def test_model_creation():
    """Test model creation."""
    try:
        from config.base_config import ExperimentConfig
        from models.model_factory import ModelFactory
        from transformers import AutoTokenizer
        
        # Create minimal config
        config = ExperimentConfig()
        config.model.model_name = "gpt2"  # Use small model for testing
        config.experiment_type = "fft"
        
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        logger.info("✓ Tokenizer loaded")
        
        # Test MaskedLinear layer creation
        from models.masked_layers import MaskedLinear
        test_layer = MaskedLinear(128, 256)
        logger.info("✓ MaskedLinear layer creation works")
        
        # We won't actually load the model to save memory
        logger.info("✓ Model factory ready (skipping actual model load)")
        return True
    except Exception as e:
        logger.error(f"✗ Model test failed: {e}")
        return False

def test_data_loading():
    """Test data loading system."""
    try:
        from data.dataset_loaders import DomainDatasetLoader
        
        # Test dataset configuration
        datasets = DomainDatasetLoader.DATASET_CONFIGS
        logger.info(f"✓ Domains available: {list(datasets.keys())}")
        
        # Test format function
        example = {'question': 'What is 2+2?', 'answer': '4'}
        formatted = DomainDatasetLoader.format_gsm8k(example)
        assert 'messages' in formatted
        logger.info("✓ Data formatting works")
        
        return True
    except Exception as e:
        logger.error(f"✗ Data loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("="*50)
    logger.info("Running MFT-TPU Local Tests")
    logger.info("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Device Detection", test_device),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\nTesting {test_name}...")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    logger.info("\n" + "="*50)
    logger.info(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("✓ All tests passed! System is ready.")
        logger.info("\nNext steps:")
        logger.info("1. Create config files in configs/ directory")
        logger.info("2. Run FFT training: python scripts/train_fft.py --config configs/gemma_2b_math_fft.yaml")
        logger.info("3. Run MFT training: python scripts/train_mft.py --config configs/gemma_2b_math_mft.yaml --base_model_path <fft_model_path>")
    else:
        logger.error("✗ Some tests failed. Please fix the issues above.")
    
    return failed == 0

if __name__ == "__main__":
    sys.exit(0 if main() else 1)