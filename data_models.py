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
    normalized_texts: Dict[str, List[str]]
    tokenized_texts: Dict[str, List[List[str]]]
    tokenisation_parity: pd.DataFrame
    joint_embeddings: Optional[np.ndarray] = None
    joint_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: pd.DataFrame = field(default_factory=pd.DataFrame)


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
    effect_size: float = 0.0


@dataclass
class StatisticalAuditResult:
    """Results from statistical bias audit"""
    response_length_stats: pd.DataFrame
    vocabulary_richness: pd.DataFrame
    lexical_diversity: pd.DataFrame
    statistical_tests: List[StatisticalMetric]
    flags: List[Dict[str, Any]]
    summary: Dict[str, Any]
    outliers: List[Dict[str, Any]]
    ngram_analysis: Dict[str, pd.DataFrame]


@dataclass
class TokenisationMetric:
    """Tokenisation metrics for a language-tokeniser pair"""
    language: str
    tokeniser: str
    avg_tokens_per_sentence: float
    fertility_penalty: float
    oov_rate: float
    subword_ratio: float
    severity: str = "Low"


@dataclass
class LinguisticAuditResult:
    """Results from linguistic bias audit"""
    tokeniser_performance: Dict[str, Dict[str, TokenisationMetric]]
    trust_aware_results: List[StatisticalMetric]
    lexical_analysis: Dict[str, Any]
    flags: List[Dict[str, Any]]
    summary: Dict[str, Any]
    content_analysis: pd.DataFrame
    complexity_analysis: pd.DataFrame


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
class CrossLingualResult:
    """Results from cross-lingual evaluation"""
    sdi_matrix: pd.DataFrame
    alignment_scores: pd.DataFrame
    rca_results: List[Dict]
    error_categories: Dict[str, int]
    flags: List[Dict[str, Any]]
    summary: Dict[str, Any]
    pair_classifications: Dict[str, Dict]


@dataclass
class BiasReport:
    """Complete bias audit report"""
    experiment: str
    timestamp: str
    dataset: str
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
    execution_time_seconds: float


@dataclass
class ExperimentResult:
    """Results from a single experiment run"""
    sample_size: int
    avg_sdi: float
    bias_level: str
    total_flags: int
    execution_time: float
    avg_response_length: float
    vocabulary_richness: float
    fertility_penalty: float
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Results from dataset validation"""
    dataset_name: str
    validation_timestamp: str
    status: str  # 'PASSED', 'FAILED', 'WARNING'
    report: Optional[Dict] = None
    error: Optional[str] = None
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)