#!/usr/bin/env python3
"""
Twitter RoBERTa Fine-tuning: LoRA vs Full Fine-tuning Comparison
Author: AI Research Team
Description: Comprehensive comparison of LoRA and full fine-tuning approaches
             for Twitter RoBERTa sentiment analysis on three-class classification.
"""

import argparse
import warnings
from pathlib import Path

from config import Config
from experiments import ComparisonExperiment, OptunaTuner
from training.utils import setup_gpu_environment, setup_cache_directories, setup_wandb_login
from utils import setup_logging

warnings.filterwarnings("ignore")


def main():
    """Main entry point for the Twitter RoBERTa experiments"""
    parser = argparse.ArgumentParser(
        description="Twitter RoBERTa Fine-tuning Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode compare                    # Compare LoRA vs Full fine-tuning
  python main.py --mode lora --config config.yaml # Run LoRA only with custom config
  python main.py --mode optimize --method lora     # Optimize LoRA hyperparameters
  python main.py --mode optimize --method both     # Optimize both methods and compare
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="compare",
        choices=["compare", "lora", "full", "optimize"],
        help="Execution mode: compare methods, run single method, or optimize hyperparameters"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (uses default config if not provided)"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["lora", "full", "both"],
        help="Method to optimize (only for --mode optimize)"
    )
    
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of optimization trials (only for --mode optimize)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path (logs to console only if not provided)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Setup environment
    setup_cache_directories()
    setup_wandb_login()
    device = setup_gpu_environment()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        config = Config.from_yaml(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = Config()
        print("Using default configuration")
    
    # Initialize experiment
    experiment = ComparisonExperiment(config, device)
    
    # Execute based on mode
    if args.mode == "compare":
        print("Running Mode: Full Comparison (LoRA vs Full Fine-tuning)")
        experiment.run_full_vs_lora_comparison()
    
    elif args.mode == "lora":
        print("Running Mode: LoRA Fine-tuning Only")
        config.use_lora = True
        config.save_model = True
        config.save_dir = "lora_only_model"
        config.lora_target_modules = ["query", "value", "key", "dense"]
        experiment.run_multiple_runs(config)
    
    elif args.mode == "full":
        print("Running Mode: Full Fine-tuning Only")
        config.use_lora = False
        config.save_model = True
        config.save_dir = "full_only_model"
        experiment.run_multiple_runs(config)
    
    elif args.mode == "optimize":
        print(f"Running Mode: Hyperparameter Optimization ({args.method})")
        tuner = OptunaTuner(device)
        
        if args.method == "lora":
            tuner.optimize_lora(args.trials)
        elif args.method == "full":
            tuner.optimize_full_finetune(args.trials)
        elif args.method == "both":
            tuner.compare_optimized_models(args.trials, args.trials)
    
    print(f"Experiment mode '{args.mode}' completed successfully!")


if __name__ == "__main__":
    main()