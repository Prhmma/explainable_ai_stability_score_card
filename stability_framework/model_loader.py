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
from transformers import BatchEncoding
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Tuple, Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# Prompt templates used to fine-tune the FLAN-T5 GLUE checkpoints (FusionBench,
# fusion_bench/tasks/flan_t5_text_generation/glue_prompt_templates.py).
T5_PROMPT_TEMPLATES = {
    "sst2": {
        "input_text": "Given the sentence '{sentence}', determine the sentiment. Is it positive or negative?",
        "label_words": ["negative", "positive"],
    },
}


class T5PromptTokenizer:
    """
    Tokenizer for prompted T5 classification that reproduces training-time tokenization.

    SentencePiece tokenization depends on context (e.g. "'it" vs "' it"), so
    tokenizing the prompt and the sentence separately would feed the model
    slightly different tokens than it was trained on. This wrapper tokenizes
    the full prompted string jointly, as during fine-tuning, and returns only
    the tokens that cover the sentence (plus </s>), with offsets relative to
    the sentence. T5LabelScoringClassifier re-inserts the fixed prompt tokens,
    so prefix + sentence tokens + suffix equals the joint tokenization.

    All other attributes are delegated to the wrapped tokenizer.
    """

    def __init__(self, tokenizer: Any, template: str):
        self.tokenizer = tokenizer
        self.prefix, self.suffix = template.split("{sentence}")
        # Fixed prompt tokens, taken from a joint tokenization so their boundaries match.
        self.prefix_ids, _, self.suffix_ids = self._split("sentence")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def _split(self, sentence: str) -> Tuple[List[int], List[Tuple[int, int]], List[int]]:
        """Joint-tokenize the prompted sentence; return (prefix ids, sentence ids + offsets, suffix ids)."""
        full = self.prefix + sentence + self.suffix
        enc = self.tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        start, end = len(self.prefix), len(self.prefix) + len(sentence)
        prefix_ids, sentence, suffix_ids = [], [], []
        for token_id, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
            if b <= start:
                prefix_ids.append(token_id)
            elif a >= end:
                suffix_ids.append(token_id)
            else:
                sentence.append((token_id, (max(a, start) - start, min(b, end) - start)))
        return prefix_ids, sentence, suffix_ids

    def __call__(
        self,
        text,
        return_tensors: Optional[str] = None,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = True,
        **kwargs,
    ):
        single = isinstance(text, str)
        texts = [text] if single else [str(t) for t in text]
        all_ids, all_offsets = [], []
        for t in texts:
            _, sentence, _ = self._split(t)
            if truncation and max_length is not None:
                sentence = sentence[: max_length - 1]
            ids = [i for i, _ in sentence]
            offsets = [o for _, o in sentence]
            if add_special_tokens:
                ids.append(self.tokenizer.eos_token_id)
                offsets.append((0, 0))
            all_ids.append(ids)
            all_offsets.append(offsets)

        if padding or return_tensors is not None:
            width = max(len(ids) for ids in all_ids)
            masks = [[1] * len(ids) + [0] * (width - len(ids)) for ids in all_ids]
            all_offsets = [o + [(0, 0)] * (width - len(o)) for o in all_offsets]
            all_ids = [ids + [self.tokenizer.pad_token_id] * (width - len(ids)) for ids in all_ids]
        else:
            masks = [[1] * len(ids) for ids in all_ids]

        data = {"input_ids": all_ids, "attention_mask": masks}
        if return_offsets_mapping:
            data["offset_mapping"] = all_offsets
        if single and return_tensors is None:
            data = {k: v[0] for k, v in data.items()}
        return BatchEncoding(data, tensor_type=return_tensors)


class T5LabelScoringClassifier(torch.nn.Module):
    """
    Text classifier backed by a T5 conditional-generation model.

    Classification follows the text-to-text formulation the checkpoint was
    trained with: the input sentence is wrapped in the task prompt, and each
    class is scored by the decoder's logit for that class's label word at the
    first decoding step. Because every label word is a single token, this is
    exactly the conditional likelihood of each label sequence.

    The prompt is part of the model, not of the input: forward() receives the
    sentence tokens produced by T5PromptTokenizer (input_ids or inputs_embeds)
    and inserts the prompt's token embeddings around them. Explainers therefore attribute
    importance to sentence tokens only, while the prompt stays fixed.
    """

    def __init__(self, t5_model: Any, tokenizer: "T5PromptTokenizer", task: str = "sst2"):
        super().__init__()
        self.t5_model = t5_model
        self.config = t5_model.config
        template = T5_PROMPT_TEMPLATES[task]
        self.register_buffer("prefix_ids", torch.tensor(tokenizer.prefix_ids, dtype=torch.long), persistent=False)
        self.register_buffer("suffix_ids", torch.tensor(tokenizer.suffix_ids, dtype=torch.long), persistent=False)
        self.eos_token_id = tokenizer.eos_token_id

        label_ids = []
        for word in template["label_words"]:
            ids = tokenizer.tokenizer(word, add_special_tokens=False).input_ids
            if len(ids) != 1:
                raise ValueError(f"Label word '{word}' must be a single token, got {ids}")
            label_ids.append(ids[0])
        self.register_buffer("label_token_ids", torch.tensor(label_ids), persistent=False)
        self.config.num_labels = len(label_ids)
        self.config.id2label = dict(enumerate(template["label_words"]))
        self.config.label2id = {w: i for i, w in self.config.id2label.items()}

    @property
    def device(self) -> torch.device:
        return next(self.t5_model.parameters()).device

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.t5_model.get_input_embeddings()

    def _wrap_in_prompt(
        self, embeds: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build [prefix] + sentence tokens (without </s>) + [suffix] + </s>, per row."""
        embed = self.get_input_embeddings()
        prefix = embed(self.prefix_ids)
        suffix = embed(torch.cat([self.suffix_ids, self.suffix_ids.new_tensor([self.eos_token_id])]))
        if attention_mask is None:
            attention_mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)

        rows = []
        for row_embeds, row_mask in zip(embeds, attention_mask):
            length = int(row_mask.sum().item())
            # The tokenizer appends </s> to the sentence; the prompt suffix goes before it.
            sentence = row_embeds[: length - 1]
            rows.append(torch.cat([prefix, sentence, suffix], dim=0))

        max_len = max(r.shape[0] for r in rows)
        out = embeds.new_zeros((len(rows), max_len, embeds.shape[-1]))
        mask = attention_mask.new_zeros((len(rows), max_len))
        for i, r in enumerate(rows):
            out[i, : r.shape[0]] = r
            mask[i, : r.shape[0]] = 1
        return out, mask

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.get_input_embeddings()(input_ids)
        prompted_embeds, prompted_mask = self._wrap_in_prompt(inputs_embeds, attention_mask)

        decoder_start = self.config.decoder_start_token_id
        decoder_input_ids = torch.full(
            (prompted_embeds.shape[0], 1), decoder_start, dtype=torch.long, device=prompted_embeds.device
        )
        outputs = self.t5_model(
            inputs_embeds=prompted_embeds,
            attention_mask=prompted_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits[:, 0, :].index_select(-1, self.label_token_ids)
        return SequenceClassifierOutput(logits=logits)


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
                tokenizer = T5PromptTokenizer(tokenizer, T5_PROMPT_TEMPLATES["sst2"]["input_text"])
                model = self._load_t5_model(model_name, tokenizer)
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
    
    def _load_t5_model(self, model_name: str, tokenizer: Any) -> Any:
        """
        Load a T5 conditional-generation checkpoint as a label-scoring classifier.
        No new parameters are introduced; predictions come from the trained decoder.
        """
        base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        return T5LabelScoringClassifier(base_model, tokenizer, task="sst2")
    
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
        "name": "tanganke/flan-t5-base_glue-sst2",
        "description": "FLAN-T5-base fine-tuned on SST-2, classifying by conditional generation of the label word",
        "architecture": "t5",
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