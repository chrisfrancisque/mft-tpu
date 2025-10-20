# setup_tpu_env.py
"""
Setup TPU environment for training.
Handles TPU initialization, environment variables, and distributed setup.
"""

import os
import sys
import logging
import subprocess
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_tpu_environment():
    """Setup environment for TPU training."""

    # Check if running on TPU - DO NOT initialize XLA device before spawn!
    is_cloud_tpu = False

    # Method 1: Check TPU_NAME environment variable
    if os.environ.get('TPU_NAME') is not None:
        is_cloud_tpu = True
        logger.info("Detected TPU via TPU_NAME environment variable")

    # Method 2: Check if torch_xla is importable (but don't call xla_device yet!)
    if not is_cloud_tpu:
        try:
            import torch_xla
            # Check for TPU environment indicators without initializing XLA
            if os.path.exists('/dev/accel0') or 'TPU' in os.environ.get('ACCELERATOR_TYPE', ''):
                is_cloud_tpu = True
                logger.info("Detected TPU via system indicators")
        except:
            pass

    if is_cloud_tpu:
        logger.info("Running on TPU environment")
        
        # Install TPU-specific requirements
        try:
            import torch_xla
            logger.info("torch_xla already installed")
        except ImportError:
            logger.info("Installing torch_xla...")
            subprocess.run([
                sys.executable, "-m", "pip", "install",
                "torch-xla[tpu]", "-f",
                "https://storage.googleapis.com/libtpu-releases/index.html"
            ], check=True)
        
        # Set XLA environment variables for optimization
        os.environ['XLA_USE_BF16'] = '1'  # Use bfloat16
        os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '100000000'  # 100MB
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging
        
        # Set TPU-specific PyTorch settings
        os.environ['PYTORCH_TPU_ALLREDUCE_TIMEOUT'] = '1800'  # 30 minutes
        
        logger.info("TPU environment configured successfully")
    else:
        logger.info("Running in CPU/GPU mode")
    
    return is_cloud_tpu


def verify_tpu_setup() -> bool:
    """Verify TPU is properly set up and accessible."""
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        
        # Test tensor operation
        test_tensor = torch.randn(2, 2).to(device)
        result = test_tensor + test_tensor
        xm.mark_step()
        
        logger.info(f"TPU verified: {device}")
        logger.info(f"Number of TPU cores: {xm.xrt_world_size()}")
        return True
        
    except Exception as e:
        logger.error(f"TPU verification failed: {str(e)}")
        return False


def get_tpu_config() -> dict:
    """Get TPU configuration from environment."""
    config = {
        'tpu_name': os.environ.get('TPU_NAME'),
        'tpu_zone': os.environ.get('TPU_ZONE'),
        'tpu_project': os.environ.get('TPU_PROJECT'),
        'num_cores': 8,  # v4-8 has 8 cores
    }
    
    # Try to detect actual number of cores
    try:
        import torch_xla.core.xla_model as xm
        config['num_cores'] = xm.xrt_world_size()
    except:
        pass
    
    return config


if __name__ == "__main__":
    # Setup and verify when run directly
    is_tpu = setup_tpu_environment()
    
    if is_tpu:
        if verify_tpu_setup():
            config = get_tpu_config()
            logger.info(f"TPU Configuration: {config}")
        else:
            logger.error("TPU setup verification failed")
            sys.exit(1)
    else:
        logger.info("Running in non-TPU mode")