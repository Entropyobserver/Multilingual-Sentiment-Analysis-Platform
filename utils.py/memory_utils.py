import torch
import gc
from functools import wraps
from training.utils import log_dual_metrics


def auto_clear_memory(func):
    """Decorator to automatically clear GPU memory"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            return result
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise e
    return wrapper


def log_gpu_memory(step_name=""):
    """Decorator to log GPU memory usage"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available():
                memory_info = {
                    f"gpu_allocated_mb_{step_name}": torch.cuda.memory_allocated()/1024**2,
                    f"gpu_cached_mb_{step_name}": torch.cuda.memory_reserved()/1024**2
                }
                # Try to log if logging function is available
                try:
                    log_dual_metrics(memory_info)
                except:
                    pass
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_memory_info() -> dict:
    """Get current memory information"""
    info = {}
    
    if torch.cuda.is_available():
        info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
        info['gpu_cached'] = torch.cuda.memory_reserved() / 1024**2  # MB
        info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    return info


def clear_memory():
    """Manually clear GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()