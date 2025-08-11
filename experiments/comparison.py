import copy
import numpy as np
import logging
from typing import Dict, List
from config import Config
from models import TwitterRoBERTaModel
from training import Trainer, Evaluator, set_random_seed
from data import load_combined_data, clean_text, ThreeClassDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os

logger = logging.getLogger(__name__)


class ComparisonExperiment:
    """Comparison experiment between Full Fine-tuning and LoRA"""
    
    def __init__(self, base_config: Config, device):
        self.base_config = base_config
        self.device = device
        self.trainer = Trainer(device)
        self.evaluator = Evaluator()
    
    def run_single_experiment(self, config: Config, run_id: int = 1) -> Dict:
        """Run a single training experiment"""
        set_random_seed(config.seed + run_id)
        
        model_type = "LoRA" if config.use_lora else "Full"
        run_name = f"twitter-roberta-3class-{model_type}-run{run_id}"
        
        # Initialize tracking
        from training.utils import init_dual_tracking, log_dual_metrics, end_dual_tracking
        tracking_config = config.to_dict()
        tracking_config.update({"seed": config.seed + run_id, "run_id": run_id})
        
        init_dual_tracking("twitter-roberta-lora", "Twitter_RoBERTa_LoRA_Comparison", run_name, tracking_config)
        
        try:
            # Initialize model
            model = TwitterRoBERTaModel(config).to(self.device)
            
            # Log model parameters
            params_info = model.get_params_info()
            log_dual_metrics(params_info)
            
            if config.use_lora:
                logger.info(f"LoRA Model - Total params: {params_info['total_params']:,}, "
                           f"Trainable: {params_info['trainable_params']:,} "
                           f"({params_info['efficiency_ratio']:.4f})")
                model.print_trainable_parameters()
            else:
                logger.info(f"Full Model - Total params: {params_info['total_params']:,}")
            
            # Load and prepare data
            train_df, test_df = load_combined_data(config)
            
            train_texts = [clean_text(t, config.keep_punctuation) for t in train_df['text']]
            test_texts = [clean_text(t, config.keep_punctuation) for t in test_df['text']]
            train_labels = train_df['label'].tolist()
            test_labels = test_df['label'].tolist()
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.15, random_state=42, stratify=train_labels
            )
            
            # Create datasets
            train_dataset = ThreeClassDataset(train_texts, train_labels, model.tokenizer, 
                                            max_length=config.max_length, augment=True)
            val_dataset = ThreeClassDataset(val_texts, val_labels, model.tokenizer, max_length=config.max_length)
            test_dataset = ThreeClassDataset(test_texts, test_labels, model.tokenizer, max_length=config.max_length)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0)
            
            # Setup training
            optimizer, scheduler = self.trainer.setup_training(model, config)
            
            # Training loop
            best_val_f1 = 0
            patience_counter = 0
            train_losses, val_losses = [], []
            train_f1s, val_f1s = [], []
            
            for epoch in range(config.num_epochs):
                # Train epoch
                avg_train_loss, train_preds, train_labels_all = self.trainer.train_epoch(
                    model, train_loader, optimizer, epoch
                )
                train_metrics = self.evaluator.calculate_metrics(train_labels_all, train_preds)
                
                # Validation epoch
                avg_val_loss, val_preds, val_labels_all = self.trainer.validate_epoch(
                    model, val_loader, epoch
                )
                val_metrics = self.evaluator.calculate_metrics(val_labels_all, val_preds)
                
                # Record metrics
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                train_f1s.append(train_metrics['f1'])
                val_f1s.append(val_metrics['f1'])
                
                # Log metrics
                log_dual_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
                log_dual_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
                log_dual_metrics({"train_loss": avg_train_loss, "val_loss": avg_val_loss}, step=epoch)
                
                logger.info(f"Epoch {epoch+1}/{config.num_epochs} - "
                           f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                           f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
                
                # Early stopping and model saving
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    patience_counter = 0
                    if config.save_model:
                        save_path = f"{config.save_dir}/run_{run_id}"
                        os.makedirs(save_path, exist_ok=True)
                        model.save_model(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= config.patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                scheduler.step()
            
            # Test the model
            test_preds, test_labels_all = self.trainer.test_model(model, test_loader)
            test_metrics = self.evaluator.calculate_metrics(test_labels_all, test_preds)
            log_dual_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            
            # Generate reports
            class_report = self.evaluator.generate_classification_report(test_labels_all, test_preds)
            logger.info(f"\nClassification Report for Run {run_id}:\n{class_report}")
            
            # Visualizations
            if config.save_model:
                learning_curves_path = f"{config.save_dir}/learning_curves_run_{run_id}.png"
                self.evaluator.plot_learning_curves(train_losses, val_losses, train_f1s, val_f1s, learning_curves_path)
                
                cm_path = f"{config.save_dir}/confusion_matrix_run_{run_id}.png"
                self.evaluator.plot_confusion_matrix(test_labels_all, test_preds, cm_path)
                
                error_analysis = self.evaluator.analyze_errors(test_texts, test_labels_all, test_preds)
                log_dual_metrics({
                    "error_rate": error_analysis['error_rate'], 
                    "total_errors": error_analysis['total_errors']
                })
            
            return test_metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            end_dual_tracking()
    
    def run_multiple_runs(self, config: Config) -> List[Dict]:
        """Run multiple experiments for statistical analysis"""
        results = []
        model_type = "LoRA" if config.use_lora else "Full"
        logger.info(f"Training Twitter RoBERTa ({model_type}): {config.model_name}")
        
        for i in range(config.num_runs):
            logger.info(f"Running experiment {i+1}/{config.num_runs}")
            metrics = self.run_single_experiment(config, run_id=i+1)
            results.append(metrics)
        
        # Calculate statistics
        avg_metrics = {
            f"avg_{metric}": np.mean([r[metric] for r in results])
            for metric in results[0].keys()
        }
        std_metrics = {
            f"std_{metric}": np.std([r[metric] for r in results])
            for metric in results[0].keys()
        }
        
        logger.info(f"Twitter RoBERTa ({model_type}) Results: "
                    f"Avg F1={avg_metrics['avg_f1']:.4f}±{std_metrics['std_f1']:.4f}, "
                    f"Avg Acc={avg_metrics['avg_accuracy']:.4f}±{std_metrics['std_accuracy']:.4f}")
        
        return results
    
    def run_full_vs_lora_comparison(self) -> Dict:
        """Run comprehensive comparison between Full and LoRA fine-tuning"""
        logger.info("Starting LoRA Comparison Experiment")
        
        self.base_config.save_model = True
        self.base_config.save_dir = "twitter_roberta_comparison"
        
        # Full Fine-tuning configuration
        full_config = copy.deepcopy(self.base_config)
        full_config.use_lora = False
        
        # LoRA configuration
        lora_config = copy.deepcopy(self.base_config)
        lora_config.use_lora = True
        lora_config.lora_target_modules = ["query", "value", "key", "dense"]
        
        logger.info("--- Training Full Fine-tuning Model ---")
        full_results = self.run_multiple_runs(full_config)
        
        logger.info("--- Training LoRA Model ---")
        lora_results = self.run_multiple_runs(lora_config)
        
        # Compare results
        full_avg_f1 = np.mean([r['f1'] for r in full_results])
        full_std_f1 = np.std([r['f1'] for r in full_results])
        lora_avg_f1 = np.mean([r['f1'] for r in lora_results])
        lora_std_f1 = np.std([r['f1'] for r in lora_results])
        
        logger.info("\n" + "="*50)
        logger.info("FINAL COMPARISON RESULTS")
        logger.info("="*50)
        logger.info(f"Full Fine-tuning: F1 = {full_avg_f1:.4f} ± {full_std_f1:.4f}")
        logger.info(f"LoRA Fine-tuning: F1 = {lora_avg_f1:.4f} ± {lora_std_f1:.4f}")
        
        if lora_avg_f1 >= full_avg_f1:
            logger.info(f"LoRA outperformed Full fine-tuning by {abs(full_avg_f1 - lora_avg_f1):.4f}")
        elif lora_avg_f1 >= full_avg_f1 * 0.95:
            logger.info(f"LoRA achieves competitive performance (within 5% of Full fine-tuning)")
        else:
            logger.info(f"LoRA shows a performance gap compared to full fine-tuning.")
        
        # Generate comparison visualization
        results_dict = {"Full": full_results, "LoRA": lora_results}
        comparison_path = f"{self.base_config.save_dir}/model_comparison.png"
        self.evaluator.compare_results(results_dict, comparison_path)
        
        return {'full_results': full_results, 'lora_results': lora_results}