from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class Config:
    """Unified configuration for Twitter RoBERTa experiments"""
    
    # Model settings
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_length: int = 128
    patience: int = 3
    num_runs: int = 3
    
    # LoRA settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["query", "value"])
    lora_bias: str = "none"
    
    # Training settings
    save_model: bool = False
    save_dir: str = "twitter_roberta_models"
    keep_punctuation: bool = True
    seed: int = 42
    enable_hyperparameter_tuning: bool = False
    
    # Data settings
    samples_per_class: int = 10000
    test_samples_per_class: int = 1500
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """Load configuration from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "num_classes": 3,
            "seed": self.seed,
            "use_lora": self.use_lora,
            "model_type": "LoRA" if self.use_lora else "Full",
            "lora_r": self.lora_r if self.use_lora else None,
            "lora_alpha": self.lora_alpha if self.use_lora else None,
            "lora_dropout": self.lora_dropout if self.use_lora else None,
            "lora_target_modules": str(self.lora_target_modules) if self.use_lora else None,
        }