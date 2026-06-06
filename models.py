"""
MaHealthBiasAudit - Data Models
Dataclasses for structured data representation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd


@dataclass
class PreprocessingResult:
    """Results from preprocessing pipeline"""
    normalized_texts: Dict[str, List[str]]  # language -> list of normalized answers
    tokenized_texts: Dict[str, List[List[str]]]  # language -> list of token lists
    tokenisation_parity: pd.DataFrame
    joint_embeddings: Optional[np.ndarray] = None
    joint_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalMetric:
    """Statistical metric for a single comparison"""
    metric_name: str
    language_1: str
    language_2: str
    value: float
    p_value: Optional[float] = None
    is_significant: bool = False
    interpretation: str = ""


@dataclass
class StatisticalAuditResult:
    """Results from statistical bias audit"""
    response_length_stats: pd.DataFrame
    vocabulary_richness: pd.DataFrame
    lexical_diversity: pd.DataFrame
    statistical_tests: List[StatisticalMetric]
    flags: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class TokenisationMetric:
    """Tokenisation metrics for a language-tokeniser pair"""
    language: str
    tokeniser: str
    avg_tokens_per_sentence: float
    fertility_penalty: float
    oov_rate: float
    subword_ratio: float


@dataclass
class LinguisticAuditResult:
    """Results from linguistic bias audit"""
    tokeniser_performance: Dict[str, Dict[str, TokenisationMetric]]
    trust_aware_results: List[StatisticalMetric]
    lexical_analysis: Dict[str, Any]
    flags: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class ModelAuditResult:
    """Results from model bias audit"""
    embeddings: Dict[str, np.ndarray]
    pairwise_distances: pd.DataFrame
    pca_projection: np.ndarray
    clustering_metrics: Dict[str, Any]
    translation_consistency: Dict[str, float]
    flags: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class CrossLingualMetric:
    """Cross-lingual evaluation metric"""
    source_lang: str
    target_lang: str
    sdi_score: float
    semantic_alignment: float
    information_retention: float
    error_type: Optional[str] = None


@dataclass
class RootCauseAnalysis:
    """Root cause analysis result"""
    root_cause: str
    affected_pairs: List[tuple]
    severity: float
    examples: List[Dict[str, str]]


@dataclass
class CrossLingualResult:
    """Results from cross-lingual evaluation"""
    sdi_matrix: pd.DataFrame
    alignment_scores: pd.DataFrame
    rca_results: List[RootCauseAnalysis]
    error_categories: Dict[str, int]
    flags: List[Dict[str, Any]]
    summary: Dict[str, Any]


@dataclass
class BiasReport:
    """Complete bias audit report"""
    experiment: str
    timestamp: str
    languages: List[str]
    total_questions: int
    total_answers: int
    key_metrics: Dict[str, Any]
    sdi_ranking: Dict[str, float]
    rca_distribution: Dict[str, int]
    flags: List[Dict[str, Any]]
    statistical_summary: Dict[str, Any]
    linguistic_summary: Dict[str, Any]
    model_summary: Dict[str, Any]
    cross_lingual_summary: Dict[str, Any]
    recommendations: List[str]
    output_directory: str