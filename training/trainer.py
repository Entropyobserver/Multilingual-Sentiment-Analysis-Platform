import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import logging
from typing import Tuple, List
from utils.memory_utils import auto_clear_memory, log_gpu_memory

logger = logging.getLogger(__name__)


class Trainer:
    """Training module for Twitter RoBERTa models"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def setup_training(self, model, config):
        """Setup optimizer and scheduler"""
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=0.01
        )
        
        scheduler = LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=0.1, 
            total_iters=config.num_epochs
        )
        
        return optimizer, scheduler
    
    @auto_clear_memory
    @log_gpu_memory("training_epoch")
    def train_epoch(self, model, train_loader: DataLoader, optimizer, epoch: int) -> Tuple[float, List[int], List[int]]:
        """Train for one epoch"""
        model.configure_for_training()
        total_loss = 0
        train_preds, train_labels_all = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(predictions.cpu().tolist())
            train_labels_all.extend(batch['labels'].cpu().tolist())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss, train_preds, train_labels_all
    
    @auto_clear_memory
    @log_gpu_memory("validation_epoch")
    def validate_epoch(self, model, val_loader: DataLoader, epoch: int) -> Tuple[float, List[int], List[int]]:
        """Validate for one epoch"""
        model.configure_for_evaluation()
        val_loss = 0
        val_preds, val_labels_all = [], []
        
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = model(**batch)
                loss = self.criterion(outputs.logits, batch['labels'])
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(predictions.cpu().tolist())
                val_labels_all.extend(batch['labels'].cpu().tolist())
                
                progress_bar.set_postfix({'val_loss': loss.item()})
        
        avg_loss = val_loss / len(val_loader)
        return avg_loss, val_preds, val_labels_all
    
    def test_model(self, model, test_loader: DataLoader) -> Tuple[List[int], List[int]]:
        """Test the model"""
        model.configure_for_evaluation()
        test_preds, test_labels_all = [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                test_preds.extend(predictions.cpu().tolist())
                test_labels_all.extend(batch['labels'].cpu().tolist())
        
        return test_preds, test_labels_all