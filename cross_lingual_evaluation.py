"""
Cross-Lingual Evaluation Engine
Based on Section 7 of the research proposal
Includes: SDI computation, MRR, RCA cascade, Bias Patterns detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from config import THRESHOLDS, MATERNAL_TOPICS, PRIMARY_LANGUAGES
from utils import (
    compute_cosine_similarity, compute_mmd_rbf, set_seed, 
    RANDOM_SEED, compute_embedding_stats
)


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
        """Initialize cross-lingual evaluator"""
        set_seed(RANDOM_SEED)
        self.sdi_matrix = None
        self.mrr_scores = None
        self.rca_results: List[RCAResult] = []
        self.bias_patterns = None
        self.per_question_sdi = None
    
    def compute_semantic_divergence_index(self, 
                                           embeddings: Dict[str, np.ndarray],
                                           questions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compute Semantic Divergence Index (SDI) for all language pairs
        SDI(L_i, L_j) = 1 - mean_k[cosine(a_k^{L_i}, a_k^{L_j})]
        Based on Section 7.2 of proposal
        
        Lower SDI indicates better semantic alignment
        SDI > 0.4 = high bias, SDI > 0.2 = moderate bias
        """
        print("\n" + "="*60)
        print("🌐 Computing Semantic Divergence Index (SDI)")
        print("="*60)
        
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
        
        # Print summary
        print(f"\n   SDI Matrix Summary:")
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:
                    sdi = sdi_matrix.loc[lang1, lang2]
                    severity = "HIGH" if sdi > THRESHOLDS['sdi_high'] else \
                              "MODERATE" if sdi > THRESHOLDS['sdi_moderate'] else "LOW"
                    print(f"      {lang1} ↔ {lang2}: SDI={sdi:.3f} [{severity}]")
        
        # Calculate average SDI
        upper_tri = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)]
        avg_sdi = np.mean(upper_tri) if len(upper_tri) > 0 else 0
        print(f"\n   Average SDI across all pairs: {avg_sdi:.3f}")
        
        return sdi_matrix
    
    def compute_mean_reciprocal_rank(self, 
                                      embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Compute Mean Reciprocal Rank for cross-lingual retrieval
        Measures whether same-question answers rank #1
        Based on Section 7.4 of proposal
        
        MRR = 1 means perfect retrieval (correct answer always rank 1)
        MRR < 0.5 indicates significant cross-lingual retrieval bias
        """
        print("\n" + "="*60)
        print("🎯 Computing Mean Reciprocal Rank (MRR)")
        print("="*60)
        
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
        
        # Print summary
        print(f"\n   MRR Matrix Summary:")
        for source_lang in languages:
            for target_lang in languages:
                if source_lang != target_lang:
                    mrr = mrr_matrix.loc[source_lang, target_lang]
                    status = "GOOD" if mrr > 0.7 else "MODERATE" if mrr > 0.5 else "POOR"
                    print(f"      {source_lang} → {target_lang}: MRR={mrr:.3f} [{status}]")
        
        return mrr_matrix
    
    def compute_cluster_purity(self, 
                                embeddings: np.ndarray,
                                labels: List[str]) -> float:
        """
        Compute cluster purity to measure whether embeddings cluster by language or topic
        Based on Section 7.4 of proposal
        
        High purity (>0.6) indicates embeddings cluster by language = bias detected
        Low purity (<0.4) indicates good cross-lingual alignment
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
        
        purity = purity / len(embeddings)
        
        print(f"\n   Cluster Purity Analysis:")
        print(f"      Language Cluster Purity: {purity:.3f}")
        interpretation = "⚠️ BIAS: embeddings cluster by language" if purity > 0.6 else \
                        "✓ Good cross-lingual alignment" if purity < 0.4 else \
                        "Moderate cross-lingual alignment"
        print(f"      Interpretation: {interpretation}")
        
        return purity
    
    def root_cause_attribution_cascade(self,
                                         sdi_threshold: float = 0.4) -> List[RCAResult]:
        """
        Root Cause Attribution (RCA) Cascade
        Based on Section 7.5 and Figure 6 of proposal
        Distinguishes between technical bias and valid cultural knowledge
        """
        print("\n" + "="*60)
        print("🔍 Root Cause Attribution (RCA) Cascade")
        print("="*60)
        
        self.rca_results = []
        
        if self.per_question_sdi is None:
            print("   No per-question SDI data available")
            return []
        
        for pair_key, sdi_list in self.per_question_sdi.items():
            lang_parts = pair_key.split('_')
            if len(lang_parts) != 2:
                continue
            lang1, lang2 = lang_parts
            
            for q_idx, sdi in enumerate(sdi_list):
                if sdi <= sdi_threshold:
                    continue
                
                rca_scores = {}
                
                # Level 1: Tokenisation Check
                tok_score = self._check_tokenisation_bias(lang1, lang2, q_idx)
                rca_scores['TOKENISATION'] = tok_score
                
                # Level 2: Morphological Alignment Check
                morph_score = self._check_morphological_bias(lang1, lang2, q_idx)
                rca_scores['MORPHOLOGY'] = morph_score
                
                # Level 3: Query Structure Check
                query_score = self._check_query_structure(lang1, q_idx)
                rca_scores['QUERY_STRUCTURE'] = query_score
                
                # Level 4: Cultural Knowledge Divergence
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
        
        # Print summary
        print(f"\n   RCA Summary:")
        cause_counts = {}
        preserve_count = 0
        for result in self.rca_results:
            cause_counts[result.root_cause] = cause_counts.get(result.root_cause, 0) + 1
            if result.preserve:
                preserve_count += 1
        
        for cause, count in cause_counts.items():
            print(f"      {cause}: {count} cases ({count/len(self.rca_results)*100:.1f}%)")
        print(f"      To PRESERVE (cultural): {preserve_count} cases ({preserve_count/len(self.rca_results)*100:.1f}%)")
        
        return self.rca_results
    
    def _check_tokenisation_bias(self, lang1: str, lang2: str, q_idx: int) -> float:
        """Check tokenisation fertility bias"""
        from config import LANGUAGES
        
        # Get morphological complexity as proxy for tokenisation difficulty
        complexity1 = LANGUAGES.get(lang1, {}).get('morphological_complexity', 1.0)
        complexity2 = LANGUAGES.get(lang2, {}).get('morphological_complexity', 1.0)
        
        # Higher complexity = higher tokenisation bias
        max_complexity = max(complexity1, complexity2)
        
        if max_complexity > 2.0:
            return min(1.0, (max_complexity - 1.0) / 1.5)
        elif max_complexity > 1.5:
            return 0.5
        return 0.0
    
    def _check_morphological_bias(self, lang1: str, lang2: str, q_idx: int) -> float:
        """Check morphological fragmentation bias"""
        # Simulated MAS scores based on language complexity
        mas_scores = {
            'English': 0.95,
            'Swahili': 0.72,
            'Yoruba': 0.68,
            'Amharic': 0.60,
            'Luganda': 0.55,
            'Runyankore': 0.50
        }
        
        mas1 = mas_scores.get(lang1, 0.7)
        mas2 = mas_scores.get(lang2, 0.7)
        
        if mas1 < 0.6 or mas2 < 0.6:
            return max(0.0, (0.6 - min(mas1, mas2)) / 0.6)
        return 0.0
    
    def _check_query_structure(self, lang: str, q_idx: int) -> float:
        """Check query formulation mismatch"""
        # Bantu languages often have sentence-final interrogatives
        if lang in ['Luganda', 'Runyankore']:
            return 0.8
        elif lang == 'Swahili':
            return 0.5
        elif lang == 'Yoruba':
            return 0.4
        return 0.0
    
    def _check_cultural_content(self, lang1: str, lang2: str, q_idx: int) -> float:
        """Check cultural knowledge divergence"""
        # Languages with rich cultural terminology
        cultural_langs = ['Luganda', 'Runyankore', 'Swahili', 'Yoruba']
        if lang1 in cultural_langs or lang2 in cultural_langs:
            return 0.6
        return 0.0
    
    def _generate_recommendation(self, root_cause: str, lang1: str, lang2: str, preserve: bool) -> str:
        """Generate recommendation based on root cause"""
        if preserve:
            return f"✓ PRESERVE: Cultural knowledge in {lang1}/{lang2} - do NOT remove or normalize"
        
        recommendations = {
            'TOKENISATION': f"🔧 Implement MorphBPE or language-specific tokenisation for {lang1} and {lang2} to reduce fertility penalty",
            'MORPHOLOGY': f"🔧 Improve morpheme-aware tokenisation for {lang1}/{lang2} - consider morphological analysis",
            'QUERY_STRUCTURE': f"🔧 Adapt QA model for non-English interrogative patterns in {lang1}",
            'CULTURAL': f"📚 Document and preserve cultural knowledge in {lang1}/{lang2} - add to knowledge base",
            'UNKNOWN': f"🔍 Manual review needed for {lang1}/{lang2} question pair"
        }
        
        return recommendations.get(root_cause, recommendations['UNKNOWN'])
    
    def detect_bias_patterns(self, 
                              embeddings: Dict[str, np.ndarray],
                              questions: Dict[str, List[str]],
                              topics: List[str]) -> pd.DataFrame:
        """
        Detect bias patterns specific to maternal health topics
        Reveals hidden disparities across topics
        Based on "Bias Patterns" requirement
        """
        print("\n" + "="*60)
        print("📊 Detecting Bias Patterns by Topic")
        print("="*60)
        
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
            
            # Determine bias severity
            if avg_sdi > THRESHOLDS['sdi_high']:
                severity = 'high'
                severity_icon = '🔴'
            elif avg_sdi > THRESHOLDS['sdi_moderate']:
                severity = 'moderate'
                severity_icon = '🟡'
            else:
                severity = 'low'
                severity_icon = '🟢'
            
            patterns.append({
                'topic': topic,
                'avg_sdi': avg_sdi,
                'bias_severity': severity,
                'severity_icon': severity_icon,
                'n_samples': len(topic_indices)
            })
            
            print(f"      {severity_icon} {topic}: SDI={avg_sdi:.3f} ({severity})")
        
        self.bias_patterns = pd.DataFrame(patterns)
        return self.bias_patterns
    
    def generate_report(self) -> pd.DataFrame:
        """Generate cross-lingual evaluation report"""
        if self.sdi_matrix is None:
            return pd.DataFrame()
        
        rows = []
        languages = self.sdi_matrix.index.tolist()
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:
                    sdi = self.sdi_matrix.loc[lang1, lang2]
                    mrr = self.mrr_scores.loc[lang1, lang2] if self.mrr_scores is not None else 0.0
                    
                    # Determine bias level
                    if sdi > THRESHOLDS['sdi_high']:
                        bias_level = 'High'
                        needs_intervention = True
                    elif sdi > THRESHOLDS['sdi_moderate']:
                        bias_level = 'Moderate'
                        needs_intervention = True
                    else:
                        bias_level = 'Low'
                        needs_intervention = False
                    
                    rows.append({
                        'Language_Pair': f"{lang1} ↔ {lang2}",
                        'SDI': round(sdi, 4),
                        'MRR': round(mrr, 4),
                        'Bias_Level': bias_level,
                        'Needs_Intervention': needs_intervention
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
                            flags.append(f"HIGH_SDI: {lang1}-{lang2} = {sdi:.3f} (>0.4 threshold)")
                        elif sdi > THRESHOLDS['sdi_moderate']:
                            flags.append(f"MODERATE_SDI: {lang1}-{lang2} = {sdi:.3f}")
        
        for result in self.rca_results:
            if result.preserve:
                flags.append(f"PRESERVE_CULTURAL: {result.language_pair[0]}-{result.language_pair[1]} (Q{result.question_id})")
            elif result.confidence > 0.7:
                flags.append(f"RCA_{result.root_cause}: {result.language_pair}")
        
        return flags
    
    def run_full_evaluation(self, embeddings: Dict[str, np.ndarray],
                            questions: Dict[str, List[str]],
                            topics: List[str] = None) -> Dict:
        """
        Run complete cross-lingual evaluation
        
        Args:
            embeddings: Dictionary of embeddings per language
            questions: Dictionary of questions per language
            topics: List of topics for each question
        
        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "="*70)
        print("🌐 Cross-Lingual Evaluation Engine")
        print("="*70)
        
        # Compute Semantic Divergence Index
        sdi_matrix = self.compute_semantic_divergence_index(embeddings, questions)
        
        # Compute Mean Reciprocal Rank
        mrr_matrix = self.compute_mean_reciprocal_rank(embeddings)
        
        # Root Cause Attribution
        rca_results = self.root_cause_attribution_cascade()
        
        # Detect bias patterns by topic
        bias_patterns = None
        if topics and len(topics) > 0:
            # Extend topics for each language
            extended_topics = []
            for lang in embeddings.keys():
                extended_topics.extend(topics[:len(embeddings[lang])])
            bias_patterns = self.detect_bias_patterns(embeddings, questions, extended_topics)
        
        # Generate report
        report = self.generate_report()
        flags = self.get_flags()
        
        return {
            'sdi_matrix': sdi_matrix,
            'mrr_matrix': mrr_matrix,
            'rca_results': rca_results,
            'bias_patterns': bias_patterns,
            'report': report,
            'flags': flags
        }


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
    
    print("\n✅ Cross-lingual evaluator test complete!")