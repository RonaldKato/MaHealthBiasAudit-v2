"""
Utility functions for MaHealthBiasAudit
Based on English, Swahili, Luganda, Runyankore
"""

import numpy as np
import pandas as pd
import random
import re
import unicodedata
from typing import List, Dict, Set, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to {seed}")


def normalize_text(text: str, lang: str = 'English', preserve_tones: bool = True) -> str:
    """Normalize text with language-specific rules"""
    if not text:
        return ""
    
    text = unicodedata.normalize('NFC', text)
    
    # Preserve tones for Luganda and Runyankore
    tonal_languages = ['Luganda', 'Runyankore']
    
    if lang not in tonal_languages or not preserve_tones:
        text = text.lower()
    
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s\.\,\?\!\-áéíóúàèìòùâêîôû]', '', text)
    
    return text


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors"""
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    
    similarity = np.dot(emb1_norm, emb2_norm.T)[0, 0]
    return float(np.clip(similarity, -1.0, 1.0))


def compute_fertility_penalty(lang_tokens: List[str], eng_tokens: List[str]) -> float:
    """Compute fertility penalty: tokens in language L / tokens in English"""
    if not eng_tokens:
        return 1.0
    return len(lang_tokens) / len(eng_tokens)


def compute_oov_rate(tokens: List[str], vocabulary: Set[str]) -> float:
    """Compute Out-of-Vocabulary rate"""
    if not tokens:
        return 0.0
    oov_count = sum(1 for t in tokens if t not in vocabulary)
    return oov_count / len(tokens)


def get_morpheme_boundaries(word: str, segments: List[str]) -> Set[int]:
    """Get character-level boundary positions for segmented word"""
    boundaries = set()
    pos = 0
    for seg in segments:
        pos += len(seg)
        boundaries.add(pos)
    return boundaries


def compute_boundary_f1(predicted: Set[int], gold: Set[int]) -> float:
    """Compute F1 score for morpheme boundary prediction"""
    if not gold:
        return 1.0 if not predicted else 0.0
    
    intersection = len(predicted & gold)
    precision = intersection / max(len(predicted), 1)
    recall = intersection / len(gold)
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_jensen_shannon_divergence(p: Dict, q: Dict, base: float = 2.0) -> float:
    """Calculate Jensen-Shannon Divergence between two probability distributions"""
    all_tokens = set(p.keys()) | set(q.keys())
    
    epsilon = 1e-12
    p_aligned = np.array([p.get(token, epsilon) for token in all_tokens])
    q_aligned = np.array([q.get(token, epsilon) for token in all_tokens])
    
    p_aligned = p_aligned / np.sum(p_aligned)
    q_aligned = q_aligned / np.sum(q_aligned)
    
    m = 0.5 * (p_aligned + q_aligned)
    
    def kl_divergence(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    
    return float(np.clip(0.5 * kl_divergence(p_aligned, m) + 0.5 * kl_divergence(q_aligned, m), 0.0, 1.0))


def compute_type_token_ratio(tokens: List[str]) -> float:
    """Compute Type-Token Ratio for lexical diversity"""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_hapax_legomena(tokens: List[str]) -> Tuple[int, float]:
    """Compute hapax legomena (words appearing exactly once)"""
    token_counts = Counter(tokens)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)
    if not tokens:
        return 0, 0.0
    return hapax_count, hapax_count / len(tokens)


def extract_cultural_terms(text: str, language: str) -> List[Dict]:
    """Extract cultural terms from text based on YOUR dataset"""
    from config import CULTURAL_TERMINOLOGY
    
    terms_found = []
    text_lower = text.lower()
    
    if language in CULTURAL_TERMINOLOGY:
        for term, (category, translation, importance, is_medical) in CULTURAL_TERMINOLOGY[language].items():
            if term.lower() in text_lower:
                terms_found.append({
                    'term': term,
                    'category': category,
                    'translation': translation,
                    'cultural_importance': importance,
                    'is_medical': is_medical,
                    'preserve': is_medical or importance > 0.8
                })
    return terms_found


def classify_question_category(question: str, data_dict: Dict) -> str:
    """Classify which category a question belongs to in YOUR dataset"""
    question_lower = question.lower()
    
    category_keywords = {
        "about_healthcare_access": ["previous pregnancies", "complications", "mosquito nets", "malaria"],
        "about_medical_history_lifestyle": ["stress", "manage", "eat", "diet", "balanced"],
        "about_mental_health_support": ["emotions", "stressed", "anxious", "praying"],
        "about_recovery_baby_care": ["rest", "support at home", "mother", "concerns"],
        "about_recovery_health": ["giving birth", "fears", "pain", "fun", "relax"],
        "about_symptoms_concerns": ["unusual pain", "discomfort", "nausea", "back pain"],
        "about_traditional_cultural_practices": ["herbs", "traditional medicines", "ginger", "cultural practices"],
        "community_cultural_considerations": ["community", "beliefs", "practices", "body", "self-care"]
    }
    
    for category, keywords in category_keywords.items():
        if any(kw in question_lower for kw in keywords):
            return category
    
    return "unknown"


def save_report(report: Dict, filepath: str):
    """Save report as JSON"""
    import json
    from datetime import datetime
    
    report['generated_at'] = datetime.now().isoformat()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {filepath}")