"""
Custom loss functions for training.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_masked_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: Optional[int] = None,
    smoothing: float = 0.0
) -> torch.Tensor:
    """
    Compute masked language modeling loss with optional label smoothing.
    
    Args:
        logits: Model outputs of shape (batch_size, seq_len, vocab_size)
        labels: Target token IDs of shape (batch_size, seq_len)
        vocab_size: Size of vocabulary for label smoothing
        smoothing: Label smoothing factor (0.0 = no smoothing)
    
    Returns:
        Scalar loss value
    """
    # Shift for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    if smoothing > 0.0 and vocab_size is not None:
        # Label smoothing
        with torch.enable_grad():
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Create smoothed target distribution
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(smoothing / (vocab_size - 1))
            smooth_targets.scatter_(1, shift_labels.unsqueeze(1), 1.0 - smoothing)
            
            # Mask out padding tokens
            padding_mask = shift_labels != -100
            smooth_targets = smooth_targets * padding_mask.unsqueeze(1)
            
            # KL divergence loss
            loss = -torch.sum(smooth_targets * log_probs, dim=-1)
            loss = loss.masked_select(padding_mask).mean()
    else:
        # Standard cross-entropy
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            reduction='mean'
        )
    
    return loss


def compute_sparsity_loss(
    model,
    target_sparsity: float = 0.1,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute sparsity regularization loss for MFT.
    
    Args:
        model: Model with MaskedLinear layers
        target_sparsity: Target sparsity ratio
        temperature: Temperature for sigmoid
    
    Returns:
        Scalar sparsity loss
    """
    from models.masked_layers import MaskedLinear
    
    sparsity_loss = 0.0
    num_layers = 0
    
    for module in model.modules():
        if isinstance(module, MaskedLinear) and module.use_mask:
            # Compute soft mask using sigmoid
            soft_mask = torch.sigmoid(module.scores / temperature)
            
            # L0 regularization - penalize deviation from target sparsity
            current_sparsity = 1.0 - soft_mask.mean()
            sparsity_loss += (current_sparsity - target_sparsity) ** 2
            num_layers += 1
    
    if num_layers > 0:
        sparsity_loss = sparsity_loss / num_layers
    
    return sparsity_loss