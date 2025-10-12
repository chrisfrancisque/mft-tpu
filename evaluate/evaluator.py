"""
Main evaluation interface for model evaluation across different domains.
"""

import torch
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import time
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('..')
from utils.device_utils import get_device_manager

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    domain: str
    dataset: str
    metrics: Dict[str, float]
    num_samples: int
    timestamp: float
    model_path: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __str__(self) -> str:
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in self.metrics.items()])
        return f"{self.dataset}: {metric_str} (n={self.num_samples})"


class Evaluator:
    """Main evaluator for model evaluation across domains."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 8
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use (auto-detect if None)
            dtype: Data type for model
            batch_size: Batch size for evaluation
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.dtype = dtype
        
        # Setup device
        self.device_manager = get_device_manager()
        self.device = self.device_manager.device if device is None else device
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize domain evaluators
        self._setup_evaluators()
    
    def _load_model(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto" if str(self.device) == "cuda" else None
        )
        
        if str(self.device) != "cuda":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded. Total parameters: {total_params:,}")
    
    def _setup_evaluators(self):
        """Setup domain-specific evaluators."""
        from .metrics.math_metrics import MathEvaluator
        from .metrics.code_metrics import CodeEvaluator
        from .metrics.instruction_metrics import InstructionEvaluator
        
        self.domain_evaluators = {
            'math': MathEvaluator(self.model, self.tokenizer, self.device),
            'coding': CodeEvaluator(self.model, self.tokenizer, self.device),
            'instruction': InstructionEvaluator(self.model, self.tokenizer, self.device)
        }
    
    def evaluate(
        self,
        domain: str,
        dataset: Optional[str] = None,
        max_samples: Optional[int] = None,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate model on specified domain/dataset.
        
        Args:
            domain: Domain to evaluate ('math', 'coding', 'instruction')
            dataset: Specific dataset to evaluate (None for all in domain)
            max_samples: Maximum samples to evaluate
            save_results: Whether to save results to disk
            output_dir: Directory to save results
            
        Returns:
            List of evaluation results
        """
        if domain not in self.domain_evaluators:
            raise ValueError(f"Unknown domain: {domain}. Choose from {list(self.domain_evaluators.keys())}")
        
        evaluator = self.domain_evaluators[domain]
        
        # Get datasets for domain
        if dataset:
            datasets = [dataset]
        else:
            datasets = evaluator.get_available_datasets()
        
        results = []
        
        for dataset_name in datasets:
            logger.info(f"Evaluating on {domain}/{dataset_name}")
            
            try:
                # Run evaluation
                metrics = evaluator.evaluate(
                    dataset=dataset_name,
                    max_samples=max_samples,
                    batch_size=self.batch_size
                )
                
                # Create result
                result = EvaluationResult(
                    domain=domain,
                    dataset=dataset_name,
                    metrics=metrics,
                    num_samples=evaluator.last_num_samples,
                    timestamp=time.time(),
                    model_path=str(self.model_path)
                )
                
                results.append(result)
                logger.info(f"Results: {result}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                continue
        
        # Save results if requested
        if save_results:
            self._save_results(results, output_dir)
        
        return results
    
    def evaluate_all_domains(
        self,
        max_samples: Optional[int] = None,
        save_results: bool = True,
        output_dir: Optional[str] = None
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluate on all domains."""
        all_results = {}
        
        for domain in self.domain_evaluators.keys():
            logger.info(f"\nEvaluating domain: {domain}")
            results = self.evaluate(
                domain=domain,
                max_samples=max_samples,
                save_results=False  # We'll save all at once
            )
            all_results[domain] = results
        
        if save_results:
            self._save_results(
                [r for results in all_results.values() for r in results],
                output_dir
            )
        
        return all_results
    
    def _save_results(self, results: List[EvaluationResult], output_dir: Optional[str] = None):
        """Save evaluation results to disk."""
        if output_dir is None:
            output_dir = self.model_path / "evaluation_results"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        results_data = [r.to_dict() for r in results]
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"eval_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Also save a summary
        summary_file = output_dir / f"eval_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("="*50 + "\n\n")
            
            for domain in set(r.domain for r in results):
                f.write(f"\n{domain.upper()} Domain:\n")
                f.write("-"*30 + "\n")
                domain_results = [r for r in results if r.domain == domain]
                for result in domain_results:
                    f.write(f"  {result}\n")
        
        logger.info(f"Summary saved to: {summary_file}")