"""
MaHealthBiasAudit - Utility Functions
Helper functions for the entire pipeline
"""

import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from config import LOGS_DIR, RANDOM_SEED, BIAS_SENTENCE_CHARACTERISTICS


# ============================================================
# Logging Setup
# ============================================================

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Setup logger for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    log_file = os.path.join(LOGS_DIR, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================
# Random Seed Management
# ============================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================
# File I/O
# ============================================================

def save_report(data: Any, filepath: str) -> None:
    """Save report as JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_report(filepath: str) -> Any:
    """Load report from JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """Save data as pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """Load data from pickle"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================
# Text Processing Utilities
# ============================================================

def basic_tokenize(text: str) -> List[str]:
    """Basic whitespace tokenization"""
    return text.lower().split()


def advanced_tokenize(text: str) -> List[str]:
    """Advanced tokenization with punctuation handling"""
    import re
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.lower().split()


def compute_vocabulary_richness(tokens: List[str]) -> float:
    """Compute vocabulary richness (type-token ratio)"""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_lexical_diversity(tokens: List[str], window_size: int = 100) -> float:
    """Compute moving average type-token ratio"""
    if len(tokens) < window_size:
        return compute_vocabulary_richness(tokens)
    
    ratios = []
    for i in range(0, len(tokens) - window_size + 1, window_size // 2):
        window = tokens[i:i + window_size]
        ratios.append(len(set(window)) / len(window))
    
    return np.mean(ratios)


def compute_average_sentence_length(text: str) -> float:
    """Compute average sentence length in words"""
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return np.mean(lengths)


def compute_readability_score(text: str) -> float:
    """Compute a simple readability score"""
    words = basic_tokenize(text)
    if not words:
        return 0
    
    avg_word_len = sum(len(w) for w in words) / len(words)
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    avg_sent_len = len(words) / max(len(sentences), 1)
    
    readability = 100 - (avg_word_len * 10 + avg_sent_len)
    return max(0, min(100, readability))


# ============================================================
# Statistical Utilities
# ============================================================

def cohens_d(array1: np.ndarray, array2: np.ndarray) -> float:
    """Compute Cohen's d effect size"""
    n1, n2 = len(array1), len(array2)
    if n1 == 0 or n2 == 0:
        return 0.0
    
    mean1, mean2 = np.mean(array1), np.mean(array2)
    var1, var2 = np.var(array1, ddof=1), np.var(array2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    return (mean1 - mean2) / pooled_std


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range"""
    if len(scores) == 0 or np.std(scores) == 0:
        return scores
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for data"""
    import scipy.stats as stats
    n = len(data)
    if n < 2:
        return (0, 0)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return (mean - ci, mean + ci)


# ============================================================
# Language Mapping
# ============================================================

def get_language_name(lang_code: str) -> str:
    """Get full language name from code"""
    language_names = {
        'en': 'English',
        'sw': 'Swahili',
        'lg': 'Luganda',
        'run': 'Runyankore',
        'English': 'English',
        'Swahili': 'Swahili',
        'Luganda': 'Luganda',
        'Runyankore': 'Runyankore',
        'Runyankole-Rukiga': 'Runyankole-Rukiga'
    }
    return language_names.get(lang_code, lang_code)


def get_language_code(lang_name: str) -> str:
    """Get language code from name"""
    language_codes = {
        'English': 'en',
        'Swahili': 'sw',
        'Luganda': 'lg',
        'Runyankore': 'run',
        'Runyankole-Rukiga': 'run'
    }
    return language_codes.get(lang_name, lang_name)


# ============================================================
# Category Mapping
# ============================================================

CATEGORY_LABELS = {
    'about_healthcare_access': 'Healthcare Access',
    'about_medical_history_lifestyle': 'Medical History & Lifestyle',
    'about_mental_health_support': 'Mental Health & Support',
    'about_recovery_baby_care': 'Recovery & Baby Care',
    'about_recovery_health': 'Recovery & Health',
    'about_symptoms_concerns': 'Symptoms & Concerns',
    'about_traditional_cultural_practices': 'Traditional & Cultural Practices',
    'community_cultural_considerations': 'Community & Cultural Considerations',
}

def get_category_label(category_key: str) -> str:
    """Get human-readable category label"""
    return CATEGORY_LABELS.get(category_key, category_key)


# ============================================================
# Bias Characteristics Printer - ENHANCED
# ============================================================

def print_bias_characteristics():
    """Print enhanced bias sentence characteristics with structures"""
    print("\n" + "="*80)
    print("BIAS-PRONE SENTENCE CHARACTERISTICS")
    print("="*80)
    print("\nThis section identifies sentence patterns that are most likely to introduce bias")
    print("in cross-lingual maternal health translations.\n")
    
    for i, (category, info) in enumerate(BIAS_SENTENCE_CHARACTERISTICS.items(), 1):
        print(f"\n{i}. {category.replace('_', ' ').upper()}")
        print(f"   ────────────────────────────────────────────────")
        print(f"   Description: {info['description']}")
        print(f"   Structure: {info.get('structure', 'N/A')}")
        print(f"\n   Example:")
        for line in info['example'].split('\n'):
            print(f"   {line}")
        print(f"\n   Detection: {info['detection']}")
        print(f"   Severity: {info['severity']}")
        print(f"   Mitigation: {info['mitigation']}")
        print("-" * 80)
    
    return BIAS_SENTENCE_CHARACTERISTICS


def get_bias_characteristics() -> Dict:
    """Return bias sentence characteristics"""
    return BIAS_SENTENCE_CHARACTERISTICS


def get_bias_examples() -> Dict[str, str]:
    """Get only bias examples for quick reference"""
    examples = {}
    for category, info in BIAS_SENTENCE_CHARACTERISTICS.items():
        examples[category] = info['example']
    return examples