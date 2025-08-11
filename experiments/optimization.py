import optuna
import logging
from config import Config
from .comparison import ComparisonExperiment

logger = logging.getLogger(__name__)


class OptunaTuner:
    """Optuna-based hyperparameter optimization"""
    
    def __init__(self, device):
        self.device = device
        self.experiment = ComparisonExperiment(Config(), device)
    
    def lora_objective(self, trial) -> float:
        """Objective function for LoRA hyperparameter optimization"""
        config = Config(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            num_epochs=trial.suggest_int("num_epochs", 2, 4),
            batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
            use_lora=True,
            lora_r=trial.suggest_categorical("lora_r", [8, 16, 32]),
            lora_alpha=trial.suggest_categorical("lora_alpha", [16, 32, 64]),
            lora_dropout=trial.suggest_float("lora_dropout", 0.05, 0.2),
            lora_target_modules=["query", "value", "key", "dense"],
            num_runs=1,  # Single run per trial for efficiency
            save_model=False
        )
        
        try:
            metrics = self.experiment.run_single_experiment(config, run_id=trial.number)
            return metrics['f1']
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def full_finetune_objective(self, trial) -> float:
        """Objective function for full fine-tuning optimization"""
        config = Config(
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            num_epochs=trial.suggest_int("num_epochs", 2, 5),
            batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
            use_lora=False,
            num_runs=1,
            save_model=False
        )
        
        try:
            metrics = self.experiment.run_single_experiment(config, run_id=trial.number)
            return metrics['f1']
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    def optimize_lora(self, n_trials: int = 20) -> optuna.Study:
        """Run LoRA hyperparameter optimization"""
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        logger.info(f"Starting LoRA Optuna optimization with {n_trials} trials")
        study.optimize(self.lora_objective, n_trials=n_trials)
        
        logger.info(f"Best LoRA hyperparameters: {study.best_params}")
        logger.info(f"Best LoRA F1 score: {study.best_value:.4f}")
        
        return study
    
    def optimize_full_finetune(self, n_trials: int = 15) -> optuna.Study:
        """Run full fine-tuning hyperparameter optimization"""
        study = optuna.create_study(
            direction="maximize", 
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        logger.info(f"Starting Full fine-tuning Optuna optimization with {n_trials} trials")
        study.optimize(self.full_finetune_objective, n_trials=n_trials)
        
        logger.info(f"Best Full fine-tuning hyperparameters: {study.best_params}")
        logger.info(f"Best Full fine-tuning F1 score: {study.best_value:.4f}")
        
        return study
    
    def compare_optimized_models(self, lora_trials: int = 20, full_trials: int = 15):
        """Compare optimized LoRA and Full fine-tuning models"""
        logger.info("Starting comprehensive hyperparameter optimization comparison")
        
        # Optimize LoRA
        lora_study = self.optimize_lora(lora_trials)
        
        # Optimize Full fine-tuning
        full_study = self.optimize_full_finetune(full_trials)
        
        # Compare best results
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZED MODEL COMPARISON")
        logger.info("="*60)
        logger.info(f"Best LoRA F1: {lora_study.best_value:.4f}")
        logger.info(f"Best LoRA params: {lora_study.best_params}")
        logger.info(f"Best Full F1: {full_study.best_value:.4f}")
        logger.info(f"Best Full params: {full_study.best_params}")
        
        if lora_study.best_value > full_study.best_value:
            diff = lora_study.best_value - full_study.best_value
            logger.info(f"Optimized LoRA outperforms optimized Full fine-tuning by {diff:.4f}")
        else:
            diff = full_study.best_value - lora_study.best_value
            logger.info(f"Optimized Full fine-tuning outperforms optimized LoRA by {diff:.4f}")
        
        return {
            'lora_study': lora_study,
            'full_study': full_study
        }