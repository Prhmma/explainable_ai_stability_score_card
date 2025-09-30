"""
Dataset Loading Module for Stability Assessment Framework

Simple, DRY implementation for loading and sampling datasets.
No hardcoded values - everything configurable via dependency injection.
"""

from datasets import load_dataset
import numpy as np
from typing import List, Dict, Tuple, Optional, NamedTuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class DatasetConfig(NamedTuple):
    """Configuration for a dataset - no hardcoded values!"""
    dataset_name: str
    split: str
    text_column: str
    label_column: str
    dataset_config: Optional[str] = None  # For datasets like glue/sst2
    display_name: Optional[str] = None


# Dataset configurations - centralized, not scattered in code
DATASET_CONFIGS = {
    "sst2": DatasetConfig(
        dataset_name="glue",
        dataset_config="sst2",
        split="validation",
        text_column="sentence",
        label_column="label",
        display_name="SST-2"
    ),
    "ag_news": DatasetConfig(
        dataset_name="ag_news",
        split="test",
        text_column="text",
        label_column="label",
        display_name="AG-News"
    )
}


class DatasetLoader:
    """
    Simple dataset loader with stratified sampling.
    
    Use load_dataset() with DATASET_CONFIGS for all loading.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize with random seed for reproducibility."""
        self.random_seed = random_seed
        np.random.seed(random_seed)
        logger.info(f"DatasetLoader initialized with seed: {random_seed}")
    
    def load_dataset(
        self,
        config: DatasetConfig,
        sample_size: Optional[int] = None,
        stratify: bool = False
    ) -> Tuple[List[str], List[int], Dict]:
        """
        Generic dataset loader using dependency injection.
        
        Args:
            config: DatasetConfig with all necessary parameters (NO hardcoding!)
            sample_size: Number of samples to return
            stratify: Whether to use stratified sampling
            
        Returns:
            (texts, labels, metadata)
        """
        display_name = config.display_name or config.dataset_name
        logger.info(f"Loading {display_name} dataset (split={config.split})")
        
        try:
            # Load dataset using injected configuration
            if config.dataset_config:
                dataset = load_dataset(config.dataset_name, config.dataset_config, split=config.split)
            else:
                dataset = load_dataset(config.dataset_name, split=config.split)
            
            # Extract texts and labels using injected column names
            texts = [example[config.text_column] for example in dataset]
            labels = [example[config.label_column] for example in dataset]
            
            # Apply sampling if requested
            if sample_size is not None and sample_size < len(texts):
                texts, labels = self._sample_data(texts, labels, sample_size, stratify)
            
            # Build metadata
            metadata = self._build_metadata(
                texts=texts,
                labels=labels,
                dataset_name=config.dataset_name.replace("_", ""),
                split=config.split if not stratify else f"{config.split}_stratified",
                text_column=config.text_column,
                label_column=config.label_column
            )
            
            logger.info(f"Loaded {len(texts)} {display_name} samples")
            logger.info(f"Class distribution: {metadata['class_distribution']}")
            
            return texts, labels, metadata
            
        except Exception as e:
            logger.error(f"Error loading {display_name} dataset: {str(e)}")
            raise
    

    def _sample_data(
        self,
        texts: List[str],
        labels: List[int],
        sample_size: int,
        stratify: bool
    ) -> Tuple[List[str], List[int]]:
        """
        Sample data with optional stratification.
        
        Args:
            texts: All texts
            labels: All labels
            sample_size: Target sample size
            stratify: If True, sample equally from each class
            
        Returns:
            (sampled_texts, sampled_labels)
        """
        if not stratify:
            # Simple random sampling
            indices = np.random.choice(len(texts), sample_size, replace=False)
            return [texts[i] for i in indices], [labels[i] for i in indices]
        
        # Stratified sampling
        num_classes = len(set(labels))
        samples_per_class = sample_size // num_classes
        
        sampled_texts = []
        sampled_labels = []
        
        for class_label in sorted(set(labels)):
            # Get indices for this class
            class_indices = [i for i, label in enumerate(labels) if label == class_label]
            
            # Sample from this class
            n_samples = min(samples_per_class, len(class_indices))
            sampled_indices = np.random.choice(class_indices, n_samples, replace=False)
            
            sampled_texts.extend([texts[i] for i in sampled_indices])
            sampled_labels.extend([labels[i] for i in sampled_indices])
        
        # Shuffle combined results
        combined = list(zip(sampled_texts, sampled_labels))
        np.random.shuffle(combined)
        sampled_texts, sampled_labels = zip(*combined)
        
        return list(sampled_texts), list(sampled_labels)
    
    def _build_metadata(
        self,
        texts: List[str],
        labels: List[int],
        dataset_name: str,
        split: str,
        text_column: str,
        label_column: str
    ) -> Dict:
        """
        Build metadata dictionary (DRY principle).
        
        Consolidates metadata creation logic in one place.
        """
        return {
            "dataset_name": dataset_name,
            "split": split,
            "total_samples": len(texts),
            "num_classes": len(set(labels)),
            "class_distribution": dict(Counter(labels)),
            "text_column": text_column,
            "label_column": label_column,
            "average_length": float(np.mean([len(text.split()) for text in texts]))
        }