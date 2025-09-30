"""
Model Loading Module for Stability Assessment Framework

Handles loading and managing different HuggingFace model architectures
with proper device management and compatibility checking.
"""

import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    T5ForConditionalGeneration, T5Tokenizer,
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    DistilBertForSequenceClassification, DistilBertTokenizer
)
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class T5ClassificationWrapper:
    """
    Wrapper for T5 models to handle classification tasks.
    T5 is a seq2seq model but needs to be adapted for classification.
    """
    
    def __init__(self, t5_model):
        self.t5_model = t5_model
        self.config = t5_model.config
        self.device = next(t5_model.parameters()).device
        
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        """
        Forward pass adapted for classification.
        """
        # Handle the case where embeddings are passed directly (for explainers like IntegratedGradients)
        if inputs_embeds is not None:
            # For T5, when using input embeddings directly, we need to go through the encoder
            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            # Use the last encoder hidden state and pool it
            last_hidden_state = encoder_outputs.last_hidden_state
            # Mean pooling over sequence length
            pooled = last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            
            # Create classification head
            if not hasattr(self, '_classification_head'):
                hidden_size = pooled.size(-1)
                self._classification_head = torch.nn.Linear(hidden_size, 2).to(self.device)
                torch.nn.init.normal_(self._classification_head.weight, std=0.02)
                torch.nn.init.zeros_(self._classification_head.bias)
            
            logits = self._classification_head(pooled)
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits)
        
        # Check if input_ids is None
        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided for T5ClassificationWrapper")
        
        # For T5 classification, we need decoder inputs
        # Create a simple decoder input (start of sequence token)
        batch_size = input_ids.size(0)
        decoder_start_token_id = self.t5_model.config.decoder_start_token_id or 0
        
        # Create decoder input ids (just the start token for each sample)
        decoder_input_ids = torch.full(
            (batch_size, 1), 
            decoder_start_token_id, 
            dtype=torch.long, 
            device=self.device
        )
        
        # Forward pass with decoder inputs
        outputs = self.t5_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            **kwargs
        )
        
        # For T5, we need to use the decoder's last hidden states
        # The output is a Seq2SeqLMOutput object with decoder_hidden_states
        if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
            # Use the last layer's hidden states
            last_hidden_state = outputs.decoder_hidden_states[-1]
        else:
            # Fallback: use the logits and project them through encoder
            # Get encoder outputs for classification
            encoder_outputs = self.t5_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Use the last encoder hidden state and pool it
            last_hidden_state = encoder_outputs.last_hidden_state
            # Mean pooling over sequence length
            pooled = last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            
            # Create classification head and return early
            if not hasattr(self, '_classification_head'):
                hidden_size = pooled.size(-1)
                self._classification_head = torch.nn.Linear(hidden_size, 2).to(self.device)
                torch.nn.init.normal_(self._classification_head.weight, std=0.02)
                torch.nn.init.zeros_(self._classification_head.bias)
            
            logits = self._classification_head(pooled)
            from types import SimpleNamespace
            return SimpleNamespace(logits=logits)
        
        # Take the first token (corresponding to the start token) and project to class logits
        pooled = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Create a simple linear projection to 2 classes (for SST-2)
        if not hasattr(self, '_classification_head'):
            hidden_size = pooled.size(-1)
            self._classification_head = torch.nn.Linear(hidden_size, 2).to(self.device)
            # Initialize with small random weights
            torch.nn.init.normal_(self._classification_head.weight, std=0.02)
            torch.nn.init.zeros_(self._classification_head.bias)
        
        logits = self._classification_head(pooled)
        
        # Return in the format expected by the rest of the code
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def eval(self):
        self.t5_model.eval()
        if hasattr(self, '_classification_head'):
            self._classification_head.eval()
        return self
    
    def train(self, mode=True):
        self.t5_model.train(mode)
        if hasattr(self, '_classification_head'):
            self._classification_head.train(mode)
        return self
    
    def requires_grad_(self, requires_grad=True):
        self.t5_model.requires_grad_(requires_grad)
        if hasattr(self, '_classification_head'):
            self._classification_head.requires_grad_(requires_grad)
        return self
    
    def to(self, device):
        self.t5_model = self.t5_model.to(device)
        self.device = device
        if hasattr(self, '_classification_head'):
            self._classification_head = self._classification_head.to(device)
        return self
        
    def parameters(self):
        """Return model parameters as an iterator."""
        for param in self.t5_model.parameters():
            yield param
        if hasattr(self, '_classification_head'):
            for param in self._classification_head.parameters():
                yield param
    
    def modules(self):
        """Return model modules as an iterator."""
        for module in self.t5_model.modules():
            yield module
        if hasattr(self, '_classification_head'):
            yield self._classification_head
    
    def named_parameters(self, prefix='', recurse=True):
        """Return named parameters."""
        for name, param in self.t5_model.named_parameters(prefix=prefix, recurse=recurse):
            yield name, param
        if hasattr(self, '_classification_head'):
            for name, param in self._classification_head.named_parameters(prefix=f'{prefix}classification_head.' if prefix else 'classification_head.', recurse=recurse):
                yield name, param
    
    @property
    def __class__(self):
        # Return the original model class for type checking
        return self.t5_model.__class__


class ModelLoader:
    """
    Unified model loader for different transformer architectures.
    Handles device management and provides consistent interface.
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            device: Target device ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = self._get_device(device)
        logger.info(f"ModelLoader initialized with device: {self.device}")
        
    def _get_device(self, device: Optional[str] = None) -> str:
        """Determine the best available device."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            # Check GPU memory for RTX 3070 constraints
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU memory available: {gpu_memory:.1f} GB")
            return "cuda"
        else:
            logger.warning("CUDA not available, using CPU")
            return "cpu"
    
    def load_model_and_tokenizer(
        self, 
        model_name: str, 
        custom_model_class: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer with automatic architecture detection.
        
        Args:
            model_name: HuggingFace model identifier
            custom_model_class: Optional custom model class to use
            
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Load tokenizer first, with special handling for LoRA/fine-tuned models
            tokenizer = self._load_tokenizer(model_name)
            
            # Handle special cases and architectures
            if "t5" in model_name.lower():
                model = self._load_t5_model(model_name)
            elif custom_model_class is not None:
                model = custom_model_class.from_pretrained(model_name)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Successfully loaded {model.__class__.__name__}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def _load_tokenizer(self, model_name: str) -> Any:
        """
        Load tokenizer with fallback for LoRA/fine-tuned models.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            Loaded tokenizer
        """
        try:
            # Try loading the tokenizer directly
            return AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer directly for {model_name}: {str(e)}")
            
            # Check if it's a LoRA model and try base model tokenizers
            if "lora" in model_name.lower() or "flan-t5" in model_name.lower():
                logger.info("Attempting to use base model tokenizer for LoRA/fine-tuned model")
                base_models = [
                    "google/flan-t5-base",
                    "t5-base",
                    "google/t5-v1_1-base"
                ]
                
                for base_model in base_models:
                    try:
                        logger.info(f"Trying base tokenizer: {base_model}")
                        tokenizer = AutoTokenizer.from_pretrained(base_model)
                        logger.info(f"Successfully loaded base tokenizer: {base_model}")
                        return tokenizer
                    except Exception as base_e:
                        logger.warning(f"Failed to load base tokenizer {base_model}: {str(base_e)}")
                        continue
            
            # If all else fails, re-raise the original exception
            raise e
    
    def _load_t5_model(self, model_name: str) -> Any:
        """
        Special handling for T5 models for classification.
        T5 requires custom setup for classification tasks.
        """
        # Load the T5 conditional generation model
        base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # Create a wrapper that handles classification
        wrapper = T5ClassificationWrapper(base_model)
        
        return wrapper
    
    def get_model_info(self, model: Any, tokenizer: Any) -> Dict[str, Any]:
        """
        Extract model information for reporting.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
            
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'vocab_size': len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size,
            'max_length': tokenizer.model_max_length,
            'device': str(model.device),
            'dtype': str(next(model.parameters()).dtype)
        }
        
        # Add architecture-specific info
        if hasattr(model.config, 'num_labels'):
            info['num_labels'] = model.config.num_labels
            
        if hasattr(model.config, 'hidden_size'):
            info['hidden_size'] = model.config.hidden_size
            
        return info
    
    def prepare_inputs(
        self, 
        texts: list, 
        tokenizer: Any, 
        max_length: int = 512,
        return_attention_mask: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare text inputs for model inference.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            return_attention_mask: Whether to return attention mask
            
        Returns:
            Dictionary with tokenized inputs
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize inputs
        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_attention_mask=return_attention_mask
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        return encoding
    
    def get_predictions(
        self, 
        model: Any, 
        inputs: Dict[str, torch.Tensor],
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get model predictions with proper handling of different architectures.
        
        Args:
            model: Loaded model
            inputs: Tokenized inputs
            return_logits: Whether to return raw logits
            
        Returns:
            Tuple of (predictions, logits or probabilities)
        """
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Handle different model outputs
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                # For models that don't have .logits attribute
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            if return_logits:
                return predictions, logits
            else:
                return predictions, probabilities
    
    def clear_memory(self):
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")


# Predefined model configurations for the four experiments
EXPERIMENT_MODELS = {
    "experiment_1": {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "description": "DistilBERT fine-tuned on SST-2",
        "architecture": "distilbert",
    },
    "experiment_2": {
        "name": "tanganke/flan-t5-base_glue-sst2",  # Would need fine-tuning for SST-2
        "description": "T5-small for SST-2 classification",
        "architecture": "t5",
        "note": "Requires custom fine-tuning setup"
    },
    "experiment_3": {
        "name": "textattack/roberta-base-ag-news",
        "description": "RoBERTa fine-tuned on AG-News",
        "architecture": "roberta",
    },
    "experiment_4": {
        "name": "jerrybuilds/distilbert-ag_news",  # Would need fine-tuning for AG-News
        "description": "DistilBERT for AG-News classification",
        "architecture": "distilbert",
        "note": "Requires custom fine-tuning setup"
    }
}