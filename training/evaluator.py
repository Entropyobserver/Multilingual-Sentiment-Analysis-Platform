import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    precision_recall_fscore_support, confusion_matrix
)
from typing import List, Dict
import wandb


class Evaluator:
    """Evaluation and visualization utilities"""
    
    @staticmethod
    def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def generate_classification_report(y_true: List[int], y_pred: List[int]) -> str:
        """Generate detailed classification report"""
        target_names = ['Negative', 'Neutral', 'Positive']
        return classification_report(y_true, y_pred, target_names=target_names, digits=4)
    
    @staticmethod
    def plot_learning_curves(train_losses: List[float], val_losses: List[float], 
                           train_f1s: List[float], val_f1s: List[float], 
                           save_path: str = None) -> None:
        """Plot learning curves for loss and F1 score"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # F1 curves
        ax2.plot(epochs, train_f1s, 'b-', label='Training F1', linewidth=2)
        ax2.plot(epochs, val_f1s, 'r-', label='Validation F1', linewidth=2)
        ax2.set_title('Training and Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            wandb.log({"learning_curves": wandb.Image(save_path)})
        
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: str = None) -> None:
        """Plot confusion matrix for 3-class classification"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Neutral', 'Positive'], 
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        
        plt.title('Twitter RoBERTa - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            wandb.log({"confusion_matrix": wandb.Image(save_path)})
        
        plt.close()
    
    @staticmethod
    def analyze_errors(texts: List[str], y_true: List[int], y_pred: List[int], 
                      n_examples: int = 3) -> Dict:
        """Analyze misclassified examples"""
        errors = []
        label_names = ['Negative', 'Neutral', 'Positive']
        
        for i, (text, true_label, pred_label) in enumerate(zip(texts, y_true, y_pred)):
            if true_label != pred_label:
                errors.append({
                    'text': text[:150] + '...' if len(text) > 150 else text,
                    'true_label': label_names[true_label],
                    'pred_label': label_names[pred_label]
                })
        
        # Log error examples to wandb
        if errors:
            error_data = [[e['text'], e['true_label'], e['pred_label']] for e in errors[:10]]
            table = wandb.Table(data=error_data, columns=['Text', 'True Label', 'Predicted Label'])
            wandb.log({"error_analysis": table})
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(y_true),
            'examples': errors[:n_examples]
        }
    
    @staticmethod
    def compare_results(results_dict: Dict[str, List[Dict]], save_path: str = None) -> None:
        """Compare results from multiple runs or models"""
        model_names = list(results_dict.keys())
        metrics = ['accuracy', 'f1', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            data = []
            labels = []
            
            for model_name, results in results_dict.items():
                metric_values = [r[metric] for r in results]
                data.extend(metric_values)
                labels.extend([model_name] * len(metric_values))
            
            # Box plot
            unique_labels = list(set(labels))
            box_data = [data[j] for j, label in enumerate(labels) if label in unique_labels]
            
            axes[i].boxplot([data[j:j+len(results_dict[unique_labels[0]])] 
                           for j in range(0, len(data), len(results_dict[unique_labels[0]]))],
                          labels=unique_labels)
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            wandb.log({"model_comparison": wandb.Image(save_path)})
        
        plt.close()