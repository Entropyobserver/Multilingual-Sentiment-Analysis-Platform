import torch
from torch.utils.data import Dataset
from typing import List, Dict
from .preprocessing import augment_text_three_class


class ThreeClassDataset(Dataset):
    """Dataset for three-class sentiment analysis"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128, augment: bool = False):
        if augment:
            aug_texts, aug_labels = [], []
            # Augment first half of the data to maintain balance
            for text, label in zip(texts[:len(texts)//2], labels[:len(labels)//2]):
                aug_samples = augment_text_three_class(text, label)
                aug_texts.extend(aug_samples)
                aug_labels.extend([label] * len(aug_samples))
            
            texts.extend(aug_texts)
            labels.extend(aug_labels)
            
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_tensors='pt'
        )
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.labels)