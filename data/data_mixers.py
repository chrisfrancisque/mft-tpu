# data/data_mixers.py
"""
Data mixing utilities for combining multiple datasets with specified proportions.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from typing import Dict, List, Optional, Union
import numpy as np
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class MixedDataset(Dataset):
    """Dataset that mixes multiple datasets with specified proportions or absolute counts."""

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        mixing_weights: Dict[str, Union[float, int]],
        total_samples: Optional[int] = None,
        seed: int = 42
    ):
        self.datasets = datasets
        self.mixing_weights = mixing_weights
        self.seed = seed

        # Detect if using absolute counts (integers > 1) or proportions (floats < 1)
        values = list(mixing_weights.values())
        using_absolute_counts = all(isinstance(v, int) and v > 1 for v in values)

        # Calculate samples per dataset
        if using_absolute_counts:
            # Use absolute counts directly (paper's approach)
            self.samples_per_dataset = {}
            for name, count in mixing_weights.items():
                # Don't exceed available data
                n_samples = min(count, len(datasets[name]))
                self.samples_per_dataset[name] = n_samples
                if n_samples < count:
                    logger.warning(f"Dataset {name} has only {n_samples} samples, requested {count}")
        else:
            # Use weights/proportions (original behavior)
            # Normalize mixing weights
            total_weight = sum(mixing_weights.values())
            self.normalized_weights = {
                name: weight / total_weight
                for name, weight in mixing_weights.items()
            }

            if total_samples is None:
                # Use all available data
                self.samples_per_dataset = {
                    name: len(dataset)
                    for name, dataset in datasets.items()
                }
            else:
                # Sample according to weights
                self.samples_per_dataset = {}
                for name, weight in self.normalized_weights.items():
                    n_samples = int(total_samples * weight)
                    # Don't exceed available data
                    n_samples = min(n_samples, len(datasets[name]))
                    self.samples_per_dataset[name] = n_samples
        
        # Create index mapping
        self._create_index_mapping()
        
        logger.info(f"Created mixed dataset with {len(self)} total samples")
        for name, n_samples in self.samples_per_dataset.items():
            logger.info(f"  {name}: {n_samples} samples ({n_samples/len(self)*100:.1f}%)")
    
    def _create_index_mapping(self):
        """Create mapping from global index to (dataset_name, local_index)."""
        self.index_mapping = []
        
        # Create indices for each dataset
        np.random.seed(self.seed)
        
        for name, dataset in self.datasets.items():
            n_samples = self.samples_per_dataset[name]
            dataset_size = len(dataset)
            
            # Sample indices (with replacement if needed)
            if n_samples <= dataset_size:
                indices = np.random.choice(dataset_size, n_samples, replace=False)
            else:
                indices = np.random.choice(dataset_size, n_samples, replace=True)
            
            for idx in indices:
                self.index_mapping.append((name, idx))
        
        # Shuffle the combined indices
        np.random.shuffle(self.index_mapping)
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        dataset_name, local_idx = self.index_mapping[idx]
        item = self.datasets[dataset_name][local_idx]
        
        # Add dataset source to the item
        item['dataset_source'] = dataset_name
        
        return item


class DataMixer:
    """Handles mixing of datasets for training."""
    
    @staticmethod
    def create_mixed_dataloader(
        datasets: Dict[str, Dataset],
        mixing_config: Dict[str, float],
        batch_size: int,
        total_samples: Optional[int] = None,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        seed: int = 42
    ) -> DataLoader:
        """Create a DataLoader with mixed datasets."""
        
        # Create mixed dataset
        mixed_dataset = MixedDataset(
            datasets=datasets,
            mixing_weights=mixing_config,
            total_samples=total_samples,
            seed=seed
        )
        
        # Create dataloader
        dataloader = DataLoader(
            mixed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True  # For TPU efficiency
        )
        
        return dataloader
    
    @staticmethod
    def create_domain_dataloaders(
        config,
        tokenizer,
        domain: str
    ) -> Dict[str, DataLoader]:
        """Create train and validation dataloaders for a domain."""
        
        from .dataset_loaders import DomainDatasetLoader, ChatDataset
        
        # Load raw datasets
        dataset_names = list(config.data.dataset_mixer.keys()) if config.data.dataset_mixer else None
        raw_datasets = DomainDatasetLoader.load_dataset_for_domain(
            domain=domain,
            dataset_names=dataset_names,
            cache_dir=config.data.cache_dir
        )
        
        if not raw_datasets:
            raise ValueError(f"No datasets loaded for domain {domain}")
        
        # Convert to ChatDataset format
        chat_datasets = {}
        for name, raw_dataset in raw_datasets.items():
            # Convert HF dataset to list of examples
            examples = [raw_dataset[i] for i in range(len(raw_dataset))]
            
            chat_dataset = ChatDataset(
                examples=examples,
                tokenizer=tokenizer,
                max_length=config.model.max_seq_length,
                mask_inputs=True
            )
            chat_datasets[name] = chat_dataset
        
        # Calculate batch size for TPU
        if config.tpu.use_tpu:
            total_batch_size = config.tpu.batch_size_per_core * config.tpu.num_tpu_cores
        else:
            total_batch_size = config.tpu.batch_size_per_core
        
        # Create train dataloader
        mixing_weights = config.data.dataset_mixer or {name: 1.0 for name in chat_datasets.keys()}
        
        train_dataloader = DataMixer.create_mixed_dataloader(
            datasets=chat_datasets,
            mixing_config=mixing_weights,
            batch_size=total_batch_size,
            total_samples=config.data.max_train_samples,
            shuffle=True,
            num_workers=config.data.dataloader_num_workers,
            pin_memory=config.data.dataloader_pin_memory,
            seed=config.training.seed
        )
        
        # For validation, use a subset of the training data (you might want separate validation sets)
        val_samples = min(config.data.max_eval_samples or 1000, len(train_dataloader.dataset) // 10)
        val_indices = random.sample(range(len(train_dataloader.dataset)), val_samples)
        val_dataset = Subset(train_dataloader.dataset, val_indices)
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=total_batch_size,
            shuffle=False,
            num_workers=config.data.dataloader_num_workers,
            pin_memory=config.data.dataloader_pin_memory,
            drop_last=False
        )
        
        logger.info(f"Created dataloaders - Train: {len(train_dataloader)} batches, Val: {len(val_dataloader)} batches")
        
        return {
            'train': train_dataloader,
            'validation': val_dataloader
        }