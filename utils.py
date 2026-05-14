"""
Utility functions for MaHealthBiasAudit v2
"""

import numpy as np
import pandas as pd
import torch
import random
import re
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np

RANDOM_SEED = 42

DIALECT_MARKERS = {
    "luganda": {
        "dialect_terms": ["ssebo", "nyabo"]
    }
}

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

def compute_cosine_similarity(a, b):
    pass

def compute_mmd_rbf(x, y):
    pass

RANDOM_SEED = 42

DIALECT_MARKERS = {
    "luganda": {
        "dialect_terms": ["ssebo", "nyabo", "oli otya"]
    },
    "swahili": {
        "dialect_terms": ["habari", "sawa", "mambo"]
    },
    "kinyarwanda": {
        "dialect_terms": ["amakuru", "yego"]
    },
    "english": {
        "dialect_terms": []
    }
}

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✅ Random seed set to {seed}")


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings"""
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    
    dot_product = np.dot(emb1, emb2.T)
    norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    
    similarity = (dot_product / (norm1 @ norm2.T))[0, 0]
    return float(similarity)


def compute_pairwise_similarities(embeddings_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute pairwise cosine similarities between language embeddings"""
    languages = list(embeddings_dict.keys())
    n = len(languages)
    sim_matrix = pd.DataFrame(np.ones((n, n)), index=languages, columns=languages)
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i < j:
                sim = compute_cosine_similarity(embeddings_dict[lang1], embeddings_dict[lang2])
                sim_matrix.loc[lang1, lang2] = sim
                sim_matrix.loc[lang2, lang1] = sim
    
    return sim_matrix


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute Maximum Mean Discrepancy with RBF kernel"""
    X = np.array(X)
    Y = np.array(Y)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    def rbf_kernel(x, y, sigma):
        diff = x[:, np.newaxis, :] - y[np.newaxis, :, :]
        sq_distances = np.sum(diff ** 2, axis=2)
        return np.exp(-sq_distances / (2 * sigma ** 2))
    
    K_xx = rbf_kernel(X, X, sigma)
    K_yy = rbf_kernel(Y, Y, sigma)
    K_xy = rbf_kernel(X, Y, sigma)
    
    mmd_sq = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    return np.sqrt(max(0, mmd_sq))


def calculate_jsd(p: Dict, q: Dict) -> float:
    """
    Calculate Jensen-Shannon Divergence between two probability distributions.
    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = (P+Q)/2
    """
    # Get union of all tokens
    all_tokens = set(p.keys()) | set(q.keys())
    
    # Create aligned probability vectors
    p_aligned = np.array([p.get(token, 1e-12) for token in all_tokens])
    q_aligned = np.array([q.get(token, 1e-12) for token in all_tokens])
    
    # Normalize
    p_aligned = p_aligned / np.sum(p_aligned)
    q_aligned = q_aligned / np.sum(q_aligned)
    
    # Midpoint distribution
    m = 0.5 * (p_aligned + q_aligned)
    m = np.maximum(m, 1e-12)
    
    # Calculate KL divergences
    def kl_divergence(a, b):
        return np.sum(a * np.log2(a / b))
    
    kl_pm = kl_divergence(p_aligned, m)
    kl_qm = kl_divergence(q_aligned, m)
    
    return float(0.5 * (kl_pm + kl_qm))


def normalize_text(text: str, lang: str = 'English') -> str:
    """Normalize text for comparison with language-specific rules"""
    import unicodedata
    
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Lowercase for Latin-script languages
    if lang in ['English', 'Swahili', 'Luganda', 'Runyankore', 'Yoruba']:
        text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but preserve important ones
    text = re.sub(r'[^\w\s\.\,\?\!\-]', '', text)
    
    return text


def extract_medical_terms(text: str) -> List[str]:
    """Extract medical terms from text"""
    medical_dictionary = {
        'folic acid', 'iron', 'calcium', 'protein', 'iodine', 'omega-3',
        'pregnancy', 'pregnant', 'labor', 'contractions', 'cervical',
        'breastfeeding', 'postpartum', 'depression', 'vaccination',
        'bcg', 'polio', 'measles', 'anemia', 'nutrition', 'prenatal',
        'midwife', 'obstetric', 'antenatal', 'postnatal', 'immunization'
    }
    
    words = text.lower().split()
    return [w for w in words if w in medical_dictionary]


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
    """Get character-level boundary positions for segments"""
    boundaries = set()
    pos = 0
    for seg in segments:
        pos += len(seg)
        boundaries.add(pos)
    return boundaries


def compute_boundary_f1(predicted: Set[int], gold: Set[int]) -> float:
    """Compute F1 score for boundary prediction"""
    if not gold:
        return 1.0 if not predicted else 0.0
    
    intersection = len(predicted & gold)
    precision = intersection / max(len(predicted), 1)
    recall = intersection / len(gold)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def detect_code_switching(text: str, language_probs: Dict[str, float], threshold: float = 0.3) -> bool:
    """Detect potential code-switching in text"""
    max_prob = max(language_probs.values()) if language_probs else 0
    return max_prob < threshold


def compute_chrf(candidate: str, reference: str, order: int = 6) -> float:
    """Compute chrF++ score between candidate and reference"""
    def get_ngrams(text, n):
        text = ' ' + text + ' '
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    precisions = []
    recalls = []
    
    for n in range(1, order + 1):
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        
        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        matches = sum(min(cand_counts[ng], ref_counts.get(ng, 0)) for ng in cand_counts)
        
        precision = matches / len(cand_ngrams) if cand_ngrams else 0
        recall = matches / len(ref_ngrams) if ref_ngrams else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    
    if avg_precision + avg_recall == 0:
        return 0.0
    
    beta_sq = 4  # beta=2 for chrF++
    return (1 + beta_sq) * (avg_precision * avg_recall) / (beta_sq * avg_precision + avg_recall)


def classify_maternal_topic(text: str) -> str:
    """Classify text into maternal health topic"""
    from config import MATERNAL_TOPICS
    
    text_lower = text.lower()
    scores = {}
    
    for topic, topic_info in MATERNAL_TOPICS.items():
        score = 0
        for keyword in topic_info['keywords']:
            if keyword in text_lower:
                score += 1
        scores[topic] = score
    
    if max(scores.values()) == 0:
        return 'general'
    
    return max(scores, key=scores.get)