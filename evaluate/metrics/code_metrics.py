"""
Evaluation metrics for coding domain (HumanEval, HumanEval+).
"""

import torch
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import json
import tempfile
import subprocess
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class CodeEvaluator:
    """Evaluator for code generation tasks."""
    
    DATASETS = {
        'humaneval': {
            'path': 'openai_humaneval',
            'entry_point_key': 'entry_point',
            'test_key': 'test',
            'prompt_key': 'prompt'
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
        batch_size: int = 1,  # Code generation typically done one at a time
        k: int = 1  # Number of samples per problem for pass@k
    ) -> Dict[str, float]:
        """Evaluate on a code dataset."""
        if dataset not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # For HumanEval, we'll use a simplified evaluation
        # In production, you'd want to use the official evaluation harness
        
        from datasets import load_dataset
        
        config = self.DATASETS[dataset]
        data = load_dataset(config['path'], split='test')
        
        if max_samples:
            data = data.select(range(min(max_samples, len(data))))
        
        self.last_num_samples = len(data)
        
        passed = 0
        total = 0
        
        for item in tqdm(data, desc=f"Evaluating {dataset}"):
            prompt = item[config['prompt_key']]
            
            # Generate code
            generated_code = self._generate_code(prompt)
            
            # Extract function from generated code
            func_code = self._extract_function(generated_code, item[config['entry_point_key']])
            
            # Test the function (simplified - actual testing would run in sandbox)
            test_passed = self._test_code(
                func_code,
                item[config['test_key']],
                item[config['entry_point_key']]
            )
            
            if test_passed:
                passed += 1
            total += 1
        
        pass_at_1 = passed / total if total > 0 else 0.0
        
        return {
            'pass@1': pass_at_1,
            'passed': passed,
            'total': total
        }
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code completion."""
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
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated = self.tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated
    
    def _extract_function(self, code: str, entry_point: str) -> str:
        """Extract the function from generated code."""
        # Simple extraction - look for the function definition
        lines = code.split('\n')
        func_lines = []
        in_function = False
        indent_level = None
        
        for line in lines:
            if f"def {entry_point}" in line:
                in_function = True
                func_lines = [line]
                # Determine indent level
                indent_level = len(line) - len(line.lstrip())
            elif in_function:
                if line.strip() == '':
                    func_lines.append(line)
                elif line.startswith(' ' * (indent_level + 1) if indent_level else ' '):
                    func_lines.append(line)
                else:
                    break
        
        return '\n'.join(func_lines)
    
    def _test_code(self, func_code: str, test_code: str, entry_point: str) -> bool:
        """Test the generated code (simplified version)."""
        # In production, this should run in a secure sandbox
        # Here we'll do a very basic check
        
        try:
            # Create a temporary Python file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the function
                f.write(func_code)
                f.write('\n\n')
                # Write the test
                f.write(test_code)
                f.write('\n\n')
                # Add a check
                f.write(f"check({entry_point})\n")
                temp_file = f.name
            
            # Run the test (UNSAFE - only for demo)
            # In production, use a proper sandbox like Docker
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Clean up
            os.unlink(temp_file)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.debug(f"Code execution failed: {e}")
            return False