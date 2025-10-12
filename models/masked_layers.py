
"""
Masked layers for MFT implementation.
Implements learnable binary masks with straight-through gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for binary mask gradient computation."""
    
    @staticmethod
    def forward(ctx, scores: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Convert scores to binary mask keeping top (1-sparsity) weights."""
        # Get the threshold for top-k selection
        k = int((1 - sparsity) * scores.numel())
        if k == 0:
            return torch.zeros_like(scores)
        
        # Get threshold value
        threshold = torch.topk(scores.flatten(), k, sorted=False).values.min()
        
        # Create binary mask
        mask = (scores >= threshold).float()
        
        # Save for backward
        ctx.save_for_backward(mask)
        
        return mask
    
    @staticmethod
    def backward(ctx, grad_output):
        """Pass gradient straight through."""
        # Gradient passes through unchanged (straight-through estimator)
        return grad_output, None


class MaskedLinear(nn.Module):
    """Linear layer with learnable mask for MFT."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sparsity: float = 0.1,
        init_method: str = "kaiming",
        init_std: float = 0.01,
        device=None,
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Original weight (frozen during MFT)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        
        # Learnable scores for mask generation
        self.scores = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        
        # Binary mask (not a parameter, computed from scores)
        self.register_buffer('mask', torch.ones((out_features, in_features), **factory_kwargs))
        
        # Flag to enable/disable masking
        self.use_mask = False
        
        # Initialize parameters
        self.reset_parameters(init_method, init_std)
    
    def reset_parameters(self, init_method: str = "kaiming", init_std: float = 0.01):
        """Initialize weights and scores."""
        # Initialize weight using standard methods
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Initialize scores for mask learning
        if init_method == "kaiming":
            nn.init.kaiming_normal_(self.scores, mode='fan_out', nonlinearity='relu')
            self.scores.data *= init_std
        elif init_method == "xavier":
            nn.init.xavier_normal_(self.scores)
            self.scores.data *= init_std
        elif init_method == "normal":
            nn.init.normal_(self.scores, mean=0.0, std=init_std)
        else:
            # Default: small random values
            nn.init.normal_(self.scores, mean=0.0, std=init_std)
    
    def compute_mask(self) -> torch.Tensor:
        """Compute binary mask from scores using straight-through estimator."""
        if self.training and self.use_mask:
            # During training, use straight-through estimator
            mask = StraightThroughEstimator.apply(self.scores, self.sparsity)
        else:
            # During inference or when mask is disabled, use fixed mask
            mask = self.mask
        
        return mask
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional masking."""
        if self.use_mask:
            # Apply mask to weights
            mask = self.compute_mask()
            masked_weight = self.weight * mask
        else:
            masked_weight = self.weight
        
        return F.linear(input, masked_weight, self.bias)
    
    def enable_mask(self):
        """Enable masking (for MFT training)."""
        self.use_mask = True
        # Freeze original weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        # Enable score learning
        self.scores.requires_grad = True
    
    def disable_mask(self):
        """Disable masking (for regular fine-tuning)."""
        self.use_mask = False
        # Enable weight training
        self.weight.requires_grad = True
        if self.bias is not None:
            self.bias.requires_grad = True
        # Disable score learning
        self.scores.requires_grad = False
    
    def apply_mask_permanently(self):
        """Apply the learned mask permanently to weights."""
        with torch.no_grad():
            # Compute final mask
            k = int((1 - self.sparsity) * self.scores.numel())
            if k > 0:
                threshold = torch.topk(self.scores.flatten(), k, sorted=False).values.min()
                final_mask = (self.scores >= threshold).float()
            else:
                final_mask = torch.zeros_like(self.scores)
            
            # Apply mask to weights permanently
            self.weight.data *= final_mask
            
            # Store final mask
            self.mask.data = final_mask
            
            # Disable further masking
            self.use_mask = False
            
            # Log sparsity statistics
            actual_sparsity = 1.0 - (final_mask.sum() / final_mask.numel())
            logger.info(f"Applied permanent mask with sparsity: {actual_sparsity:.4f}")
    
    def get_sparsity_stats(self) -> dict:
        """Get current sparsity statistics."""
        with torch.no_grad():
            if self.use_mask:
                mask = self.compute_mask()
            else:
                mask = self.mask
            
            total_params = mask.numel()
            active_params = mask.sum().item()
            sparsity = 1.0 - (active_params / total_params)
            
            return {
                'total_params': total_params,
                'active_params': active_params,
                'sparsity': sparsity,
                'target_sparsity': self.sparsity
            }


def replace_linear_with_masked(
    model: nn.Module,
    config,
    target_layers: Optional[list] = None
) -> nn.Module:
    """Replace Linear layers with MaskedLinear layers in specified model layers."""
    
    if target_layers is None:
        target_layers = config.mask.masked_layers
    
    def replace_in_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this module should be masked
            layer_idx = None
            for idx in target_layers:
                if f"layer.{idx}" in full_name or f"layers.{idx}" in full_name:
                    layer_idx = idx
                    break
            
            if layer_idx is not None:
                # Check if it's a Linear layer to replace
                if isinstance(child, nn.Linear):
                    should_replace = False
                    
                    # Check if it's attention or MLP
                    if config.mask.apply_to_attention and any(
                        key in full_name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']
                    ):
                        should_replace = True
                    
                    if config.mask.apply_to_mlp and any(
                        key in full_name for key in ['gate_proj', 'up_proj', 'down_proj', 'mlp', 'fc']
                    ):
                        should_replace = True
                    
                    if should_replace:
                        # Create masked linear layer
                        masked_layer = MaskedLinear(
                            child.in_features,
                            child.out_features,
                            bias=child.bias is not None,
                            sparsity=config.mask.sparsity_ratio,
                            init_method=config.mask.score_init_method,
                            init_std=config.mask.score_init_std,
                            device=child.weight.device,
                            dtype=child.weight.dtype
                        )
                        
                        # Copy weights
                        with torch.no_grad():
                            masked_layer.weight.copy_(child.weight)
                            if child.bias is not None:
                                masked_layer.bias.copy_(child.bias)
                        
                        # Replace the layer
                        setattr(module, name, masked_layer)
                        logger.info(f"Replaced Linear layer at {full_name} with MaskedLinear")
                else:
                    # Recursively replace in children
                    replace_in_module(child, full_name)
            else:
                # Recursively check children even if not in target layers
                replace_in_module(child, full_name)
    
    replace_in_module(model)
    return model