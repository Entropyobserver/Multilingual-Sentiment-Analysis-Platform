import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base_model import BaseModel
from .adapters import create_lora_model, get_model_params_info
from config import Config
from typing import Dict, Any


class TwitterRoBERTaModel(BaseModel):
    """Unified TwitterRoBERTa model wrapper supporting both full fine-tuning and LoRA"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name, 
            num_labels=3
        )
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self.model = create_lora_model(self.model, self.config)
    
    def forward(self, **kwargs):
        """Forward pass through the model"""
        return self.model(**kwargs)
    
    def save_model(self, save_path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path: str):
        """Load model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        if self.config.use_lora:
            # Handle LoRA model loading
            base_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name, num_labels=3
            )
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, load_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
    
    def get_params_info(self) -> Dict[str, Any]:
        """Get model parameters information"""
        return get_model_params_info(self.model)
    
    def configure_for_training(self):
        """Set model to training mode"""
        self.model.train()
    
    def configure_for_evaluation(self):
        """Set model to evaluation mode"""
        self.model.eval()
    
    def to(self, device):
        """Move model to specified device"""
        self.model = self.model.to(device)
        return self
    
    def parameters(self):
        """Get model parameters"""
        return self.model.parameters()
    
    def named_parameters(self):
        """Get named model parameters"""
        return self.model.named_parameters()
    
    def print_trainable_parameters(self):
        """Print trainable parameters (useful for LoRA)"""
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            params_info = self.get_params_info()
            print(f"Trainable parameters: {params_info['trainable_params']:,} / {params_info['total_params']:,}")