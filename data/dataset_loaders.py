# data/dataset_loaders.py
"""
Dataset loaders for different domains (math, coding, instruction).
Handles loading, preprocessing, and formatting of various datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from typing import Dict, List, Optional, Union, Any
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class ChatDataset(Dataset):
    """Generic dataset for chat-formatted training data."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer,
        max_length: int = 2048,
        mask_inputs: bool = True
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_inputs = mask_inputs
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format messages into chat template
        text = self.tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Create labels (mask input if specified)
        labels = input_ids.clone()
        if self.mask_inputs:
            # Mask everything except assistant responses
            labels = self._mask_inputs(example['messages'], input_ids, labels)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _mask_inputs(self, messages, input_ids, labels):
        """Mask non-assistant parts of the conversation."""
        # This is a simplified version - you might need to adjust based on your tokenizer
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels


class DomainDatasetLoader:
    """Loads and preprocesses datasets for specific domains."""
    
    # Dataset configurations for each domain
    DATASET_CONFIGS = {
        'math': {
            'gsm8k': {
                'path': 'gsm8k',
                'split': 'train',
                'subset': 'main',
                'format_fn': 'format_gsm8k'
            },
            'math_qa': {
                'path': 'math_qa',
                'split': 'train',
                'format_fn': 'format_math_qa'
            },
            'metamath': {
                'path': 'meta-math/MetaMathQA',
                'split': 'train',
                'format_fn': 'format_metamath'
            }
        },
        'coding': {
            'humaneval': {
                'path': 'openai_humaneval',
                'split': 'train',
                'format_fn': 'format_humaneval'
            },
            'code_alpaca': {
                'path': 'sahil2801/CodeAlpaca-20k',
                'split': 'train',
                'format_fn': 'format_code_alpaca'
            },
            'evol_code': {
                'path': 'nickrosh/Evol-Instruct-Code-80k-v1',
                'split': 'train',
                'format_fn': 'format_evol_code'
            },
            # Paper's dataset for replication
            'tulu3_persona_python': {
                'path': 'allenai/tulu-3-sft-personas-code',
                'split': 'train',
                'format_fn': 'format_tulu3_persona_python'
            }
        },
        'instruction': {
            'alpaca': {
                'path': 'tatsu-lab/alpaca',
                'split': 'train',
                'format_fn': 'format_alpaca'
            },
            'openassistant': {
                'path': 'OpenAssistant/oasst1',
                'split': 'train',
                'format_fn': 'format_openassistant'
            },
            'dolly': {
                'path': 'databricks/databricks-dolly-15k',
                'split': 'train',
                'format_fn': 'format_dolly'
            }
        }
    }
    
    @classmethod
    def load_dataset_for_domain(
        cls,
        domain: str,
        dataset_names: Optional[List[str]] = None,
        cache_dir: str = "./data_cache"
    ) -> Dict[str, HFDataset]:
        """Load datasets for a specific domain."""
        
        if domain not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown domain: {domain}. Choose from {list(cls.DATASET_CONFIGS.keys())}")
        
        domain_configs = cls.DATASET_CONFIGS[domain]
        
        if dataset_names is None:
            dataset_names = list(domain_configs.keys())
        
        loaded_datasets = {}
        
        for dataset_name in dataset_names:
            if dataset_name not in domain_configs:
                logger.warning(f"Dataset {dataset_name} not configured for domain {domain}")
                continue
            
            config = domain_configs[dataset_name]
            logger.info(f"Loading {dataset_name} for {domain} domain")
            
            try:
                # Load dataset
                dataset = load_dataset(
                    config['path'],
                    config.get('subset'),
                    split=config['split'],
                    cache_dir=cache_dir
                )
                
                # Format dataset
                format_fn_name = config['format_fn']
                format_fn = getattr(cls, format_fn_name)
                formatted_dataset = dataset.map(format_fn, remove_columns=dataset.column_names)
                
                loaded_datasets[dataset_name] = formatted_dataset
                logger.info(f"Loaded {len(formatted_dataset)} examples from {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {str(e)}")
                continue
        
        return loaded_datasets
    
    # Dataset formatting functions
    @staticmethod
    def format_gsm8k(example):
        """Format GSM8K dataset."""
        question = example['question']
        answer = example['answer']
        
        messages = [
            {"role": "user", "content": f"Solve this math problem: {question}"},
            {"role": "assistant", "content": answer}
        ]
        return {"messages": messages}
    
    @staticmethod
    def format_math_qa(example):
        """Format Math QA dataset."""
        problem = example.get('problem', example.get('question', ''))
        solution = example.get('solution', example.get('answer', ''))
        
        messages = [
            {"role": "user", "content": f"Solve this problem: {problem}"},
            {"role": "assistant", "content": solution}
        ]
        return {"messages": messages}
    
    @staticmethod
    def format_metamath(example):
        """Format MetaMath dataset."""
        query = example.get('query', '')
        response = example.get('response', '')
        
        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ]
        return {"messages": messages}
    
    @staticmethod
    def format_humaneval(example):
        """Format HumanEval dataset."""
        prompt = example.get('prompt', '')
        canonical_solution = example.get('canonical_solution', '')
        
        messages = [
            {"role": "user", "content": f"Complete this Python function:\n{prompt}"},
            {"role": "assistant", "content": canonical_solution}
        ]
        return {"messages": messages}
    
    @staticmethod
    def format_code_alpaca(example):
        """Format Code Alpaca dataset."""
        instruction = example.get('instruction', '')
        output = example.get('output', '')
        
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        return {"messages": messages}
    
    @staticmethod
    def format_evol_code(example):
        """Format Evol Code dataset."""
        instruction = example.get('instruction', '')
        output = example.get('output', '')

        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
        return {"messages": messages}

    @staticmethod
    def format_tulu3_persona_python(example):
        """Format Tulu 3 Persona Python dataset."""
        # Tulu 3 already has messages format
        if 'messages' in example:
            return {"messages": example['messages']}
        else:
            # Fallback if format is different
            instruction = example.get('instruction', example.get('prompt', ''))
            output = example.get('output', example.get('response', ''))

            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            return {"messages": messages}

    @staticmethod
    def format_alpaca(example):
        """Format Alpaca dataset."""
        instruction = example.get('instruction', '')
        input_text = example.get('input', '')
        output = example.get('output', '')
        
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}"
        else:
            prompt = instruction
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output}
        ]
        return {"messages": messages}
    
    @staticmethod
    def format_openassistant(example):
        """Format OpenAssistant dataset."""
        # OpenAssistant has a more complex structure
        text = example.get('text', '')
        role = example.get('role', 'user')
        
        # Simple formatting - you might need to handle conversation threads
        messages = [
            {"role": role, "content": text}
        ]
        return {"messages": messages}
    
    @staticmethod
    def format_dolly(example):
        """Format Dolly dataset."""
        instruction = example.get('instruction', '')
        context = example.get('context', '')
        response = example.get('response', '')
        
        if context:
            prompt = f"{instruction}\n\nContext: {context}"
        else:
            prompt = instruction
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return {"messages": messages}