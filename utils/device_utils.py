"""
TPU and device utilities for Training
Handles device detection, initialization, and distributed setup
"""

import os
import torch
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# check to see if TPU is available
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    TPU_AVAILABLE= True
except:
    TPU_AVAILABLE = False
    logger.warning("TPU Libraries not available. Running in CPU/GPU mode")


@dataclass
class DeviceConfig:
    """Configuration for device setup"""
    device_type: str
    device: Any
    is_distributed: bool
    world_size: int
    rank: int
    is_main_process: bool

class DeviceManager:
    """Manages device setup and provides unified interface for TPU/GPU/CPU"""

    def __init__(self, force_cpu: bool = False):
        self.force_cpu = force_cpu
        self.device_config = self._initialize_device()

    def _initialize_device(self) -> DeviceConfig:
        """Detect and initialize the appropriate device"""

        #Force CPU if requested
        if self.force_cpu:
            logger.info("Forcing CPU mode")
            return DeviceConfig(
                device_type = 'cpu',
                device = torch.device('cpu'),
                is_distributed= False,
                world_size = 1,
                rank = 0,
                is_main_process = True
            
            )
        
        # Checking TPU availablitiy
        if TPU_AVAILABLE and self._is_tpu_available():
            return self._setup_tpu()
        
        # Checking CUDA availability
        if torch.cuda.is_available():
            return self._setup_cuda()
        
        return self._setup_cpu()
    
    def _is_tpu_available(self) -> bool:
        """Check if TPU is actually available"""
        if not TPU_AVAILABLE:
            return False

        # Don't call xla_device() here - it will be called later in _setup_tpu()
        # Just check if we're in a TPU environment
        try:
            # Check for TPU indicators without initializing XLA
            return (os.path.exists('/dev/accel0') or
                    'TPU' in os.environ.get('ACCELERATOR_TYPE', '') or
                    os.environ.get('TPU_NAME') is not None)
        except:
            return False
        
    def _setup_tpu(self) -> DeviceConfig:
        """Setup TPU device configuration"""
        logger.info("Initializing TPU")

        device = xm.xla_device()

        # Use new runtime API instead of deprecated xrt_world_size/get_ordinal
        try:
            import torch_xla.runtime as xr
            world_size = xr.world_size()
            rank = xr.global_ordinal()
        except (ImportError, AttributeError):
            # Fallback to old API if new one not available
            world_size = xm.xrt_world_size()
            rank = xm.get_ordinal()

        is_main = rank == 0

        logger.info(f"TPU initialized: rank {rank}/{world_size}")

        return DeviceConfig(
            device_type = 'tpu',
            device = device,
            is_distributed = world_size > 1,
            world_size= world_size,
            rank = rank,
            is_main_process= is_main
        )
    
    def _setup_cuda(self) -> DeviceConfig:
        """Setup CUDA device configuration"""
        logger.info("Initializing CUDA")

        #Check for distributed setup
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            rank = int(os.environ.get("RANK", 0))

            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)

            logger.info(f"CUDA distributed: rank {rank}/{world_size}")

            return DeviceConfig(
                device_type= 'cuda',
                device = device,
                is_distributed= world_size > 1,
                world_size=world_size,
                rank = rank, 
                is_main_process = rank == 0
            )
        else:
            #sigle GPU setup
            device = torch.device('cuda:0')
            logger.info("CUDA single GPU mode")

            return DeviceConfig(
                device_type= 'cuda',
                device = device,
                is_distributed=False,
                world_size=1,
                rank = 0, 
                is_main_process = True
            )
    
    def _setup_cpu(self) -> DeviceConfig:
        """Setup CPU device configuration"""
        logger.info("Initializing CPU")

        return DeviceConfig(
            device_type = 'cpu',
            device = torch.device('cpu'),
            is_distributed=False,
            world_size=1,
            rank = 0,
            is_main_process= True
        )
    
    @property
    def device(self):
        """Get the device object"""
        return self.device_config.device
    
    @property
    def is_tpu(self) -> bool:
        """Check if running on TPU."""
        return self.device_config.device_type == 'tpu'
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.device_config.is_main_process
    
    def synchronize(self):
        """Synchronize across all devices"""
        if self.is_tpu:
            xm.mark_step()
        elif self.device_config.device_type == 'cuda':
            torch.cuda.synchronize()
    
    def print_once(self, *args, **kwargs):
        """Print only on the main process"""
        if self.is_main_process:
            print(*args, **kwargs)
    
    def save_checkpoint(self, model: torch.nn.Module, path: str, optimizer = None, **kwargs):
        """Save checkpoint with device-appropriate method."""
        if self.is_main_process:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                **kwargs
            }
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
            if self.is_tpu:
                xm.save(checkpoint, path)
            else:
                torch.save(checkpoint, path)
        
            logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str, map_location = None):
        """Load checkpoint with device-appropriate mathod"""
        if self.is_tpu:
            return torch.load(path, map_location='cpu')
        else:
            return torch.load(path, map_location=map_location or self.device)
    

_device_manager: Optional[DeviceManager] = None

def get_device_manager(force_cpu: bool = False) -> DeviceManager:
    """Get or create the global device manager"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(force_cpu=force_cpu)
    
    return _device_manager

def get_device():
    """helper to get the current device"""
    return get_device_manager().device


