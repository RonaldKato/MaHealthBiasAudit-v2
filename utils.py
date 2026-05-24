"""
Utility functions for MaHealthBiasAudit v2
Common help (functions) used across all modules
"""

import numpy as np
import pandas as pd
import random
import re
import unicodedata
from typing import List, Dict, Set, Tuple, Optional, Union
from collections import Counter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import torch, but don't fail if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import RANDOM_SEED, MATERNAL_TOPICS


def set_seed(seed: int = RANDOM_SEED):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def normalize_text(text: str, lang: str = 'English', preserve_tones: bool = True) -> str:
    """
    Normalize text with language-specific rules
    
    Args:
        text: Input text string
        lang: Language code/name
        preserve_tones: Whether to preserve tonal diacritics for Bantu languages
    
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Unicode normalization (NFC)
    text = unicodedata.normalize('NFC', text)
    
    # Preserve tones for tonal languages if requested
    tonal_languages = ['Luganda', 'Runyankore', 'Yoruba']
    
    if lang in tonal_languages and preserve_tones:
        # Extract tone-marked characters temporarily
        tones = re.findall(r'[áéíóúàèìòùâêîôû]', text)
        text = text.lower()
        # Tones are preserved in the Unicode NFC form
    else:
        text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-linguistic markup but preserve essential punctuation
    text = re.sub(r'[^\w\s\.\,\?\!\-áéíóúàèìòùâêîôû]', '', text)
    
    return text


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors
    
    Args:
        emb1: First embedding (1D or 2D)
        emb2: Second embedding (1D or 2D)
    
    Returns:
        Cosine similarity score between 0 and 1
    """
    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)
    
    # Normalize embeddings
    emb1_norm = emb1 / (np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-8)
    
    similarity = np.dot(emb1_norm, emb2_norm.T)[0, 0]
    return float(np.clip(similarity, -1.0, 1.0))


def compute_pairwise_similarities(embeddings_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute pairwise cosine similarities between language embeddings
    
    Args:
        embeddings_dict: Dictionary mapping language names to embedding matrices
    
    Returns:
        DataFrame of pairwise similarities
    """
    languages = list(embeddings_dict.keys())
    n = len(languages)
    sim_matrix = pd.DataFrame(np.ones((n, n)), index=languages, columns=languages)
    
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i < j:
                emb1 = embeddings_dict[lang1]
                emb2 = embeddings_dict[lang2]
                
                # Compute average similarity across all pairs
                min_len = min(len(emb1), len(emb2))
                sims = []
                for k in range(min_len):
                    sim = compute_cosine_similarity(emb1[k:k+1], emb2[k:k+1])
                    sims.append(sim)
                
                avg_sim = np.mean(sims) if sims else 0.0
                sim_matrix.loc[lang1, lang2] = avg_sim
                sim_matrix.loc[lang2, lang1] = avg_sim
    
    return sim_matrix


def compute_jensen_shannon_divergence(p: Dict, q: Dict, base: float = 2.0) -> float:
    """
    Calculate Jensen-Shannon Divergence between two probability distributions
    
    Args:
        p: First probability distribution (dict of token: probability)
        q: Second probability distribution (dict of token: probability)
        base: Logarithm base (2 for bits, e for nats)
    
    Returns:
        JSD value between 0 and 1
    """
    # Get union of all tokens
    all_tokens = set(p.keys()) | set(q.keys())
    
    # Create aligned probability vectors with smoothing
    epsilon = 1e-12
    p_aligned = np.array([p.get(token, epsilon) for token in all_tokens])
    q_aligned = np.array([q.get(token, epsilon) for token in all_tokens])
    
    # Normalize
    p_aligned = p_aligned / np.sum(p_aligned)
    q_aligned = q_aligned / np.sum(q_aligned)
    
    # Midpoint distribution
    m = 0.5 * (p_aligned + q_aligned)
    
    def kl_divergence(a, b):
        """Kullback-Leibler divergence"""
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    
    # Calculate JSD
    jsd = 0.5 * kl_divergence(p_aligned, m) + 0.5 * kl_divergence(q_aligned, m)
    
    return float(np.clip(jsd, 0.0, 1.0))


def compute_mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel
    
    Args:
        X: First sample set
        Y: Second sample set
        sigma: RBF kernel bandwidth
    
    Returns:
        MMD distance
    """
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
    return float(np.sqrt(max(0, mmd_sq)))


def compute_fertility_penalty(lang_tokens: List[str], eng_tokens: List[str]) -> float:
    """
    Compute fertility penalty: tokens in language L / tokens in English
    
    Args:
        lang_tokens: List of tokens in target language
        eng_tokens: List of tokens in English
    
    Returns:
        Fertility penalty ratio (>1 means more tokens needed)
    """
    if not eng_tokens:
        return 1.0
    
    return len(lang_tokens) / len(eng_tokens)


def compute_oov_rate(tokens: List[str], vocabulary: Set[str]) -> float:
    """
    Compute Out-of-Vocabulary rate
    
    Args:
        tokens: List of tokens to check
        vocabulary: Reference vocabulary set
    
    Returns:
        OOV rate (0-1)
    """
    if not tokens:
        return 0.0
    
    oov_count = sum(1 for t in tokens if t not in vocabulary)
    return oov_count / len(tokens)


def get_morpheme_boundaries(word: str, segments: List[str]) -> Set[int]:
    """
    Get character-level boundary positions for segmented word
    
    Args:
        word: Original word
        segments: List of morpheme/token segments
    
    Returns:
        Set of boundary positions
    """
    boundaries = set()
    pos = 0
    for seg in segments:
        pos += len(seg)
        boundaries.add(pos)
    return boundaries


def compute_boundary_f1(predicted: Set[int], gold: Set[int]) -> float:
    """
    Compute F1 score for morpheme boundary prediction
    
    Args:
        predicted: Predicted boundary positions
        gold: Gold standard boundary positions
    
    Returns:
        F1 score (0-1)
    """
    if not gold:
        return 1.0 if not predicted else 0.0
    
    intersection = len(predicted & gold)
    precision = intersection / max(len(predicted), 1)
    recall = intersection / len(gold)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_chrf(candidate: str, reference: str, order: int = 6, beta: int = 2) -> float:
    """
    Compute chrF++ score between candidate and reference
    
    Args:
        candidate: Candidate string
        reference: Reference string
        order: Maximum n-gram order
        beta: Weight for F-beta (beta=2 gives chrF2)
    
    Returns:
        chrF score (0-1)
    """
    def get_ngrams(text, n):
        """Extract character n-grams"""
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
    
    beta_sq = beta ** 2
    return (1 + beta_sq) * (avg_precision * avg_recall) / (beta_sq * avg_precision + avg_recall)


def classify_maternal_topic(text: str) -> str:
    """
    Classify text into maternal health topic based on keyword matching
    
    Args:
        text: Input text
    
    Returns:
        Topic name
    """
    text_lower = text.lower()
    scores = {}
    
    for topic, topic_info in MATERNAL_TOPICS.items():
        score = 0
        for keyword in topic_info['keywords']:
            if keyword in text_lower:
                score += 1
        scores[topic] = score
    
    max_score = max(scores.values())
    if max_score == 0:
        return 'general'
    
    return max(scores, key=scores.get)


def extract_medical_terms(text: str) -> List[str]:
    """
    Extract medical terms from text
    
    Args:
        text: Input text
    
    Returns:
        List of medical terms found
    """
    medical_dictionary = {
        'folic acid', 'iron', 'calcium', 'protein', 'iodine', 'omega-3',
        'pregnancy', 'pregnant', 'labor', 'contractions', 'cervical',
        'breastfeeding', 'postpartum', 'depression', 'vaccination',
        'bcg', 'polio', 'measles', 'anemia', 'nutrition', 'prenatal',
        'midwife', 'obstetric', 'antenatal', 'postnatal', 'immunization',
        'vitamin', 'supplement', 'fetus', 'neonatal', 'maternal'
    }
    
    words = text.lower().split()
    return [w for w in words if w in medical_dictionary]


def detect_code_switching(text: str, language_probs: Dict[str, float], threshold: float = 0.3) -> bool:
    """
    Detect potential code-switching in text
    
    Args:
        text: Input text
        language_probs: Dictionary of language probabilities
        threshold: Confidence threshold
    
    Returns:
        True if code-switching detected
    """
    max_prob = max(language_probs.values()) if language_probs else 0
    return max_prob < threshold


def compute_type_token_ratio(tokens: List[str]) -> float:
    """
    Compute Type-Token Ratio for lexical diversity
    
    Args:
        tokens: List of tokens
    
    Returns:
        Type-Token Ratio (0-1)
    """
    if not tokens:
        return 0.0
    
    unique_types = len(set(tokens))
    total_tokens = len(tokens)
    
    return unique_types / total_tokens


def compute_hapax_legomena(tokens: List[str]) -> Tuple[int, float]:
    """
    Compute hapax legomena (words appearing exactly once)
    
    Args:
        tokens: List of tokens
    
    Returns:
        Tuple of (count, proportion)
    """
    token_counts = Counter(tokens)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)
    
    if not tokens:
        return 0, 0.0
    
    return hapax_count, hapax_count / len(tokens)


def standardize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    Standardize embeddings to zero mean and unit variance
    
    Args:
        embeddings: Embedding matrix
    
    Returns:
        Standardized embeddings
    """
    scaler = StandardScaler()
    return scaler.fit_transform(embeddings)


def compute_embedding_stats(embeddings: np.ndarray) -> Dict:
    """
    Compute statistics for embedding matrix
    
    Args:
        embeddings: Embedding matrix
    
    Returns:
        Dictionary of statistics
    """
    return {
        'shape': embeddings.shape,
        'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        'min_norm': float(np.min(np.linalg.norm(embeddings, axis=1))),
        'max_norm': float(np.max(np.linalg.norm(embeddings, axis=1))),
        'mean_value': float(np.mean(embeddings)),
        'std_value': float(np.std(embeddings))
    }


def save_json(data: Dict, filepath: str):
    """
    Save dictionary as JSON file
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    import json
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict:
    """
    Load JSON file as dictionary
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded dictionary
    """
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_report(report: Dict, filepath: str):
    """
    Save report as JSON with timestamp
    
    Args:
        report: Report dictionary
        filepath: Output file path
    """
    import json
    from datetime import datetime
    
    report['generated_at'] = datetime.now().isoformat()
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Report saved to: {filepath}")