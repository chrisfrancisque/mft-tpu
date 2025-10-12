"""
Evaluation metrics for math domain (GSM8K, MATH).
"""

import torch
import re
from typing import Dict, List, Optional, Any
from datasets import load_dataset
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MathEvaluator:
    """Evaluator for math reasoning tasks."""
    
    DATASETS = {
        'gsm8k': {
            'path': 'gsm8k',
            'subset': 'main',
            'split': 'test',
            'answer_key': 'answer',
            'question_key': 'question'
        },
        'math': {
            'path': 'hendrycks/competition_math',
            'split': 'test',
            'answer_key': 'solution',
            'question_key': 'problem'
        }
    }
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.last_num_samples = 0
    
    def get_available_datasets(self) -> List[str]:
        return list(self.DATASETS.keys())
    
    def evaluate(
        self,
        dataset: str,
        max_samples: Optional[int] = None,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """Evaluate on a math dataset."""
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        config = self.DATASETS[dataset]
        
        # Load dataset
        logger.info(f"Loading {dataset} dataset")
        if 'subset' in config:
            data = load_dataset(config['path'], config['subset'], split=config['split'])
        else:
            data = load_dataset(config['path'], split=config['split'])
        
        # Limit samples if specified
        if max_samples:
            data = data.select(range(min(max_samples, len(data))))
        
        self.last_num_samples = len(data)
        
        # Evaluate
        correct = 0
        total = 0
        
        for i in tqdm(range(0, len(data), batch_size), desc=f"Evaluating {dataset}"):
            batch = data[i:i+batch_size]
            
            # Prepare prompts
            prompts = []
            for item in batch:
                question = item[config['question_key']]
                prompt = self._format_prompt(question, dataset)
                prompts.append(prompt)
            
            # Generate answers
            generated_answers = self._generate_batch(prompts)
            
            # Check answers
            for j, gen_answer in enumerate(generated_answers):
                true_answer = batch[j][config['answer_key']]
                
                # Extract numerical answer
                pred_num = self._extract_number(gen_answer)
                true_num = self._extract_number(true_answer)
                
                if pred_num is not None and true_num is not None:
                    if abs(pred_num - true_num) < 1e-5:
                        correct += 1
                
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def _format_prompt(self, question: str, dataset: str) -> str:
        """Format question into prompt."""
        if dataset == 'gsm8k':
            prompt = f"Question: {question}\n\nLet's solve this step by step:\n"
        else:  # MATH dataset
            prompt = f"Problem: {question}\n\nSolution:\n"
        
        return prompt
    
    def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate answers for a batch of prompts."""
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0,  # Greedy decoding for consistency
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated = self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        # Remove commas
        text = text.replace(',', '')
        
        # Look for patterns like "answer is X" or "= X"
        patterns = [
            r'answer is[\s:]*([+-]?\d*\.?\d+)',
            r'=\s*([+-]?\d*\.?\d+)',
            r'([+-]?\d*\.?\d+)\s*(?:is|are|was|were)\s*(?:the|our)?\s*(?:final)?\s*answer',
            r'####\s*([+-]?\d*\.?\d+)',  # GSM8K format
            r'(?:^|\s)([+-]?\d*\.?\d+)$'  # Last number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        # Try to find any number
        numbers = re.findall(r'[+-]?\d*\.?\d+', text)
        if numbers:
            try:
                # Return the last number found
                return float(numbers[-1])
            except ValueError:
                pass
        
        return None