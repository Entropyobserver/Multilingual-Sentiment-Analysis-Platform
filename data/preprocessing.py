import re
from typing import List


def clean_text(text: str, keep_punctuation: bool = True) -> str:
    """Enhanced text cleaning"""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    if not keep_punctuation:
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip().lower()


def augment_text_three_class(text: str, label: int) -> List[str]:
    """Simplified data augmentation for three-class sentiment"""
    augmented = [text]
    
    # Label-specific word replacements
    replacements = {
        0: {'bad': 'terrible', 'poor': 'awful', 'hate': 'dislike'},  # negative
        1: {'okay': 'fine', 'average': 'decent', 'normal': 'regular'},  # neutral
        2: {'good': 'great', 'nice': 'excellent', 'love': 'adore'}  # positive
    }
    
    if label in replacements:
        text_lower = text.lower()
        for old, new in replacements[label].items():
            if old in text_lower:
                augmented.append(text_lower.replace(old, new))
                break  # Only one replacement per text
    
    return augmented