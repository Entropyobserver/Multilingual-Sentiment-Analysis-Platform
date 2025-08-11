import random
import numpy as np
import torch
import os
import mlflow
import wandb
from transformers import set_seed
from typing import Dict, Any


def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_dual_tracking(project_name: str, experiment_name: str, run_name: str, config_dict: Dict[str, Any]):
    """Initialize both MLflow and wandb tracking"""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(config_dict)
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        reinit=True,
        tags=["lora", "twitter-roberta", "sentiment-analysis"]
    )


def log_dual_metrics(metrics_dict: Dict[str, Any], step: int = None):
    """Log metrics to both MLflow and wandb"""
    for key, value in metrics_dict.items():
        if step is not None:
            mlflow.log_metric(key, value, step=step)
            wandb.log({key: value}, step=step)
        else:
            mlflow.log_metric(key, value)
            wandb.log({key: value})


def log_dual_artifacts(file_path: str, artifact_name: str = None):
    """Log artifacts to both platforms"""
    mlflow.log_artifact(file_path)
    if artifact_name:
        wandb.log_artifact(file_path, name=artifact_name)
    else:
        wandb.log_artifact(file_path)


def end_dual_tracking():
    """End both tracking sessions"""
    mlflow.end_run()
    wandb.finish()


def setup_gpu_environment():
    """Setup GPU environment and memory management"""
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"Memory fraction set to: 70% of dedicated memory")
        return torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        return torch.device("cpu")


def setup_cache_directories():
    """Setup Hugging Face cache directories"""
    cache_dir = os.path.join(os.getcwd(), "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = cache_dir
    print(f"Using Hugging Face cache directory: {cache_dir}")


def setup_wandb_login():
    """Setup wandb login with environment variable"""
    wandb_api_key = os.getenv('WANDB_API_KEY', '37c342a59f2ddcfebe2b6aa3830000a1f0895758')
    if wandb_api_key:
        wandb.login(key=wandb_api_key)