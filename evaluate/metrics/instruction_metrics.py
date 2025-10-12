"""
Evaluation metrics for instruction following (IF-Eval, AlpacaEval).
"""

import torch
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import re

logger = logging.getLogger(__name__)


class InstructionEvaluator:
    """Evaluator for instruction following tasks."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.last_num_samples = 0
    
    def get_available_datasets(self) -> List[str]:
        return ['ifeval', 'alpaca_eval']
    
    def evaluate(
        self,
        dataset: str,
        max_samples: Optional[int] = None,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """Evaluate on instruction following dataset."""
        
        if dataset == 'ifeval':
            return self._evaluate_ifeval(max_samples, batch_size)
        elif dataset == 'alpaca_eval':
            return self._evaluate_alpaca(max_samples, batch_size)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def _evaluate_ifeval(
        self,
        max_samples: Optional[int] = None,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """Evaluate on IF-Eval dataset."""
        # Simplified IF-Eval - checks for instruction following constraints
        
        # Load some example constraints
        test_cases = [
            {
                'instruction': "Write a haiku about artificial intelligence. Make sure it follows the 5-7-5 syllable pattern.",
                'constraints': ['haiku_format']
            },
            {
                'instruction': "List exactly 5 benefits of exercise. Number each item.",
                'constraints': ['exact_count', 'numbered_list']
            },
            {
                'instruction': "Write a paragraph about space exploration without using the letter 'e'.",
                'constraints': ['no_letter_e']
            }
        ]
        
        if max_samples:
            test_cases = test_cases[:max_samples]
        
        self.last_num_samples = len(test_cases)
        
        total_constraints = 0
        satisfied_constraints = 0
        
        for case in tqdm(test_cases, desc="Evaluating IF-Eval"):
            response = self._generate_response(case['instruction'])
            
            for constraint in case['constraints']:
                total_constraints += 1
                if self._check_constraint(response, constraint):
                    satisfied_constraints += 1
        
        constraint_satisfaction = satisfied_constraints / total_constraints if total_constraints > 0 else 0.0
        
        return {
            'constraint_satisfaction_rate': constraint_satisfaction,
            'satisfied': satisfied_constraints,
            'total': total_constraints
        }
    
    def _evaluate_alpaca(
        self,
        max_samples: Optional[int] = None,
        batch_size: int = 4
    ) -> Dict[str, float]:
        """Evaluate on AlpacaEval dataset (simplified)."""
        # This is a simplified version - actual AlpacaEval uses GPT-4 as judge
        
        test_instructions = [
            "Explain quantum computing to a 10-year-old.",
            "Write a professional email declining a job offer.",
            "Create a recipe for a healthy breakfast smoothie.",
            "Describe the process of photosynthesis.",
            "Write a short story about a time traveler."
        ]
        
        if max_samples:
            test_instructions = test_instructions[:max_samples]
        
        self.last_num_samples = len(test_instructions)
        
        # Generate responses and score them
        scores = []
        
        for instruction in tqdm(test_instructions, desc="Evaluating AlpacaEval"):
            response = self._generate_response(instruction)
            
            # Simple quality checks
            score = self._score_response_quality(response, instruction)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return {
            'average_score': avg_score,
            'num_samples': len(scores)
        }
    
    def _generate_response(self, instruction: str) -> str:
        """Generate response to instruction."""
        # Format as chat
        messages = [
            {"role": "user", "content": instruction}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _check_constraint(self, response: str, constraint: str) -> bool:
        """Check if response satisfies a constraint."""
        if constraint == 'haiku_format':
            # Simple check for 3 lines
            lines = response.strip().split('\n')
            return len(lines) == 3
        
        elif constraint == 'exact_count':
            # Check for exactly 5 items
            numbers = re.findall(r'\d+\.', response)
            return len(numbers) == 5
        
        elif constraint == 'numbered_list':
            # Check for numbered items
            return bool(re.search(r'\d+\.', response))
        
        elif constraint == 'no_letter_e':
            # Check no 'e' or 'E'
            return 'e' not in response.lower()
        
        return False
    
    def _score_response_quality(self, response: str, instruction: str) -> float:
        """Score response quality (simplified)."""
        score = 0.0
        
        # Length check
        if 20 < len(response.split()) < 500:
            score += 0.25
        
        # Relevance check (very basic)
        instruction_words = set(instruction.lower().split())
        response_words = set(response.lower().split())
        overlap = len(instruction_words & response_words) / len(instruction_words)
        score += min(0.25, overlap)
        
        # Structure check
        if '\n' in response or '. ' in response:
            score += 0.25
        
        # Completeness check
        if response.strip() and not response.strip().endswith('...'):
            score += 0.25
        
        return min(1.0, score)