from peft import LoraConfig, get_peft_model, TaskType
from config import Config
from typing import Dict, Any


def create_lora_model(model, config: Config):
    """Create model with LoRA adapters"""
    if not config.use_lora:
        return model
    
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
    )
    
    model = get_peft_model(model, peft_config)
    return model


def get_model_params_info(model) -> Dict[str, Any]:
    """Get comprehensive model parameters information"""
    if hasattr(model, 'print_trainable_parameters'):
        # LoRA model
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
    else:
        # Full fine-tuning model
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'efficiency_ratio': trainable_params / total_params if total_params > 0 else 0,
        'parameter_reduction': 1 - (trainable_params / total_params) if total_params > 0 else 0
    }


def get_lora_config_from_config(config: Config) -> LoraConfig:
    """Convert Config to LoraConfig"""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.SEQ_CLS,
    )