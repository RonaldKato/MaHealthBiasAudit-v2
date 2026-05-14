"""
Cross-Lingual Evaluation Engine
Includes: SDI computation, MRR, RCA cascade, Bias Patterns detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.linalg import orthogonal_procrustes
import warnings
warnings.filterwarnings('ignore')

from config import THRESHOLDS, MATERNAL_TOPICS
from utils import compute_cosine_similarity, compute_mmd_rbf, set_seed, RANDOM_SEED


@dataclass
class RCAResult:
    """Root Cause Attribution result"""
    language_pair: Tuple[str, str]
    question_id: int
    sdi_value: float
    root_cause: str  # TOKENISATION, MORPHOLOGY, QUERY_STRUCTURE, CULTURAL, UNKNOWN
    confidence: float
    preserve: bool  # True for cultural knowledge that should be preserved
    recommendation: str


class CrossLingualEvaluator:
    """
    Translation-free cross-lingual evaluation engine
    Based on Section 7 of the research proposal
    """
    
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.sdi_matrix = None
        self.mrr_scores = None
        self.rca_results: List[RCAResult] = []
        self.bias_patterns = None
    
    def compute_semantic_divergence_index(self, 
                                           embeddings: Dict[str, np.ndarray],
                                           questions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compute Semantic Divergence Index (SDI) for all language pairs
        SDI(L_i, L_j) = 1 - mean_k[cosine(a_k^{L_i}, a_k^{L_j})]
        Based on Section 7.2 of proposal
        """
        languages = list(embeddings.keys())
        n = len(languages)
        sdi_matrix = pd.DataFrame(np.zeros((n, n)), index=languages, columns=languages)
        
        # Also store per-question SDI for RCA
        self.per_question_sdi = {}
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i == j:
                    sdi_matrix.loc[lang1, lang2] = 0.0
                    continue
                
                emb1 = embeddings[lang1]
                emb2 = embeddings[lang2]
                
                # Compute pairwise cosine similarities for each question
                min_len = min(len(emb1), len(emb2))
                sims = []
                per_q_sims = []
                
                for k in range(min_len):
                    sim = compute_cosine_similarity(emb1[k:k+1], emb2[k:k+1])
                    sims.append(sim)
                    per_q_sims.append(sim)
                
                mean_sim = np.mean(sims) if sims else 0.0
                sdi = 1 - mean_sim
                
                sdi_matrix.loc[lang1, lang2] = sdi
                sdi_matrix.loc[lang2, lang1] = sdi
                
                # Store per-question SDI
                pair_key = f"{lang1}_{lang2}"
                self.per_question_sdi[pair_key] = per_q_sims
        
        self.sdi_matrix = sdi_matrix
        return sdi_matrix
    
    def compute_mean_reciprocal_rank(self, 
                                      embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compute Mean Reciprocal Rank for cross-lingual retrieval
        Measures whether same-question answers rank #1 - Based on Section 7.4 of proposal
        """
        languages = list(embeddings.keys())
        mrr_matrix = pd.DataFrame(np.zeros((len(languages), len(languages))),
                                  index=languages, columns=languages)
        
        for source_lang in languages:
            for target_lang in languages:
                if source_lang == target_lang:
                    mrr_matrix.loc[source_lang, target_lang] = 1.0
                    continue
                
                source_embs = embeddings[source_lang]
                target_embs = embeddings[target_lang]
                
                ranks = []
                min_len = min(len(source_embs), len(target_embs))
                
                for query_idx in range(min_len):
                    query_emb = source_embs[query_idx:query_idx+1]
                    
                    # Compute similarities to all target answers
                    similarities = cosine_similarity(query_emb, target_embs)[0]
                    
                    # Get rank of correct answer
                    correct_sim = similarities[query_idx]
                    rank = np.sum(similarities > correct_sim) + 1
                    ranks.append(1.0 / rank)
                
                mrr = np.mean(ranks) if ranks else 0.0
                mrr_matrix.loc[source_lang, target_lang] = mrr
        
        self.mrr_scores = mrr_matrix
        return mrr_matrix
    
    def compute_cluster_purity(self, 
                                embeddings: np.ndarray,
                                labels: List[str]) -> float:
        """
        Compute cluster purity to measure whether embeddings cluster by language or topic
        Based on Section 7.4 of proposal
        """
        if len(embeddings) == 0:
            return 0.0
        
        unique_labels = list(set(labels))
        n_clusters = min(len(unique_labels), len(embeddings))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute purity
        purity = 0
        for cluster in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            
            cluster_true_labels = [labels[i] for i in cluster_indices]
            label_counts = {}
            for label in cluster_true_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            max_label_count = max(label_counts.values()) if label_counts else 0
            purity += max_label_count
        
        return purity / len(embeddings)
    
    def root_cause_attribution_cascade(self,
                                         sdi_threshold: float = 0.4) -> List[RCAResult]:
        """
        Root Cause Attribution (RCA) Cascade
        Based on Section 7.5 and Figure 6 of proposal - Distinguishes between technical bias and valid cultural knowledge
        """
        self.rca_results = []
        
        if self.per_question_sdi is None:
            return []
        
        for pair_key, sdi_list in self.per_question_sdi.items():
            lang1, lang2 = pair_key.split('_')
            
            for q_idx, sdi in enumerate(sdi_list):
                if sdi <= sdi_threshold:
                    continue
                
                rca_scores = {}
                
                # Level 1: Tokenisation Check
                # Is TP > 1.5 or OOV > 15%?
                tok_score = self._check_tokenisation_bias(lang1, lang2, q_idx)
                rca_scores['TOKENISATION'] = tok_score
                
                # Level 2: Morphological Alignment Check
                # Is MAS < 0.6?
                morph_score = self._check_morphological_bias(lang1, lang2, q_idx)
                rca_scores['MORPHOLOGY'] = morph_score
                
                # Level 3: Query Structure Check
                # Do questions have non-English interrogative constructions?
                query_score = self._check_query_structure(lang1, q_idx)
                rca_scores['QUERY_STRUCTURE'] = query_score
                
                # Level 4: Cultural Knowledge Divergence
                # Contains indigenous medical terminology?
                cultural_score = self._check_cultural_content(lang1, lang2, q_idx)
                rca_scores['CULTURAL'] = cultural_score
                
                # Determine primary root cause
                primary_cause = max(rca_scores, key=rca_scores.get)
                confidence = rca_scores[primary_cause]
                
                # Cultural knowledge should be PRESERVED, not removed
                preserve = primary_cause == 'CULTURAL' and confidence > 0.5
                
                recommendation = self._generate_recommendation(primary_cause, lang1, lang2, preserve)
                
                result = RCAResult(
                    language_pair=(lang1, lang2),
                    question_id=q_idx,
                    sdi_value=sdi,
                    root_cause=primary_cause,
                    confidence=confidence,
                    preserve=preserve,
                    recommendation=recommendation
                )
                self.rca_results.append(result)
        
        return self.rca_results
    
    def _check_tokenisation_bias(self, lang1: str, lang2: str, q_idx: int) -> float:
        """Tokenisation Fertility Bias"""
        # Simulated - would use actual TP values
        tp_values = {
            'English': 1.0,
            'Swahili': 1.6,
            'Luganda': 2.0,
            'Runyankore': 2.3,
            'Yoruba': 1.8,
            'Amharic': 2.0
        }
        
        tp1 = tp_values.get(lang1, 1.0)
        tp2 = tp_values.get(lang2, 1.0)
        
        if tp1 > 1.5 or tp2 > 1.5:
            return min(1.0, (max(tp1, tp2) - 1.0) / 1.0)
        return 0.0
    
    def _check_morphological_bias(self, lang1: str, lang2: str, q_idx: int) -> float:
        """Morphological Fragmentation"""
        # Simulated MAS scores
        mas_values = {
            'English': 0.95,
            'Swahili': 0.72,
            'Luganda': 0.58,
            'Runyankore': 0.52,
            'Yoruba': 0.65,
            'Amharic': 0.60
        }
        
        mas1 = mas_values.get(lang1, 0.8)
        mas2 = mas_values.get(lang2, 0.8)
        
        if mas1 < 0.6 or mas2 < 0.6:
            return max(0.0, (0.6 - min(mas1, mas2)) / 0.6)
        return 0.0
    
    def _check_query_structure(self, lang: str, q_idx: int) -> float:
        """Query formulation mismatch"""
        # Bantu languages often have sentence-final interrogatives
        if lang in ['Luganda', 'Runyankore']:
            return 0.8
        elif lang == 'Swahili':
            return 0.5
        return 0.0
    
    def _check_cultural_content(self, lang1: str, lang2: str, q_idx: int) -> float:
        """Cultural Knowledge Divergence"""
        # Simulated - would check actual cultural terms
        cultural_langs = ['Luganda', 'Runyankore', 'Swahili']
        if lang1 in cultural_langs or lang2 in cultural_langs:
            return 0.6
        return 0.0
    
    def _generate_recommendation(self, root_cause: str, lang1: str, lang2: str, preserve: bool) -> str:
        """Recommendation based on root cause"""
        if preserve:
            return f"PRESERVE: Cultural knowledge in {lang1}/{lang2} - do not remove"
        
        recommendations = {
            'TOKENISATION': f"Implement MorphBPE for {lang1} and {lang2} to reduce fertility penalty",
            'MORPHOLOGY': f"Improve morpheme-aware tokenisation for {lang1}/{lang2}",
            'QUERY_STRUCTURE': f"Adapt QA model for non-English interrogative patterns in {lang1}",
            'CULTURAL': f"Document and preserve cultural knowledge in {lang1}/{lang2}",
            'UNKNOWN': f"Manual review needed for {lang1}/{lang2} question pair"
        }
        
        return recommendations.get(root_cause, recommendations['UNKNOWN'])
    
    def detect_bias_patterns(self, 
                              embeddings: Dict[str, np.ndarray],
                              questions: Dict[str, List[str]],
                              topics: List[str]) -> pd.DataFrame:
        """
        Detect bias patterns specific to maternal health topics
        Reveals hidden disparities across topics - Based on "Bias Patterns" requirement
        """
        patterns = []
        
        # Get unique topics
        unique_topics = list(set(topics)) if topics else list(MATERNAL_TOPICS.keys())
        
        for topic in unique_topics:
            # Get indices for this topic
            topic_indices = [i for i, t in enumerate(topics) if t == topic] if topics else []
            
            if not topic_indices:
                continue
            
            # Compute topic-specific embeddings
            topic_embeddings = {}
            for lang, emb in embeddings.items():
                if len(emb) > max(topic_indices) if topic_indices else False:
                    topic_embeddings[lang] = emb[topic_indices]
            
            # Compute topic-specific SDI
            if len(topic_embeddings) >= 2:
                languages = list(topic_embeddings.keys())
                sdi_sum = 0
                pair_count = 0
                
                for i, lang1 in enumerate(languages):
                    for j, lang2 in enumerate(languages):
                        if i < j:
                            emb1 = topic_embeddings[lang1]
                            emb2 = topic_embeddings[lang2]
                            min_len = min(len(emb1), len(emb2))
                            
                            sims = []
                            for k in range(min_len):
                                sim = compute_cosine_similarity(emb1[k:k+1], emb2[k:k+1])
                                sims.append(sim)
                            
                            if sims:
                                sdi_sum += (1 - np.mean(sims))
                                pair_count += 1
                
                avg_sdi = sdi_sum / max(pair_count, 1)
            else:
                avg_sdi = 0.5
            
            patterns.append({
                'topic': topic,
                'avg_sdi': avg_sdi,
                'bias_severity': 'high' if avg_sdi > THRESHOLDS['sdi_high'] else
                                 'moderate' if avg_sdi > THRESHOLDS['sdi_moderate'] else 'low',
                'n_samples': len(topic_indices)
            })
        
        self.bias_patterns = pd.DataFrame(patterns)
        return self.bias_patterns
    
    def generate_report(self) -> pd.DataFrame:
        """Cross-lingual Evaluation Report"""
        if self.sdi_matrix is None:
            return pd.DataFrame()
        
        rows = []
        languages = self.sdi_matrix.index.tolist()
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:
                    sdi = self.sdi_matrix.loc[lang1, lang2]
                    mrr = self.mrr_scores.loc[lang1, lang2] if self.mrr_scores is not None else 0.0
                    
                    rows.append({
                        'Language_Pair': f"{lang1} → {lang2}",
                        'SDI': round(sdi, 4),
                        'MRR': round(mrr, 4),
                        'Bias_Level': 'High' if sdi > THRESHOLDS['sdi_high'] else
                                     'Moderate' if sdi > THRESHOLDS['sdi_moderate'] else 'Low',
                        'Needs_Intervention': sdi > THRESHOLDS['sdi_moderate']
                    })
        
        return pd.DataFrame(rows)
    
    def get_flags(self) -> List[str]:
        """Generate flags based on cross-lingual evaluation"""
        flags = []
        
        if self.sdi_matrix is not None:
            for i, lang1 in enumerate(self.sdi_matrix.index):
                for j, lang2 in enumerate(self.sdi_matrix.columns):
                    if i < j:
                        sdi = self.sdi_matrix.loc[lang1, lang2]
                        if sdi > THRESHOLDS['sdi_high']:
                            flags.append(f"HIGH_SDI: {lang1}-{lang2} = {sdi:.3f}")
        
        for result in self.rca_results:
            if result.preserve:
                flags.append(f"PRESERVE_CULTURAL: {result.language_pair[0]}-{result.language_pair[1]}")
            elif result.confidence > 0.7:
                flags.append(f"RCA_{result.root_cause}: {result.language_pair}")
        
        return flags


# Test the evaluator
if __name__ == "__main__":
    evaluator = CrossLingualEvaluator()
    
    # Sample embeddings
    np.random.seed(42)
    sample_embeddings = {
        'English': np.random.randn(5, 768),
        'Luganda': np.random.randn(5, 768) + 0.3,
        'Runyankore': np.random.randn(5, 768) + 0.5,
        'Swahili': np.random.randn(5, 768) + 0.2
    }
    
    sdi = evaluator.compute_semantic_divergence_index(sample_embeddings, {})
    print("SDI Matrix:")
    print(sdi)
    
    print("\n Cross-lingual evaluator test complete!")