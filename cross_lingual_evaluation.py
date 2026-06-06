"""
MaHealthBiasAudit - Cross-Lingual Evaluation
Cross-lingual semantic analysis and root cause identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from config import PRIMARY_LANGUAGES, SDI_THRESHOLD_HIGH, SDI_THRESHOLD_MODERATE, RCA_TOP_K
from utils import setup_logger


class CrossLingualEvaluator:
    """Cross-lingual evaluation for bias detection"""
    
    def __init__(self):
        self.logger = setup_logger('cross_lingual')
    
    def compute_semantic_distance_index(self, 
                                       embeddings: np.ndarray,
                                       labels: List[str]) -> pd.DataFrame:
        """Compute Semantic Distance Index (SDI) between language pairs"""
        if embeddings.size == 0:
            return pd.DataFrame(columns=PRIMARY_LANGUAGES, index=PRIMARY_LANGUAGES)
        
        unique_langs = list(set(labels))
        sdi_matrix = np.zeros((len(unique_langs), len(unique_langs)))
        
        for i, lang1 in enumerate(unique_langs):
            indices1 = [idx for idx, l in enumerate(labels) if l == lang1]
            if not indices1:
                continue
            
            emb1 = embeddings[indices1]
            
            for j, lang2 in enumerate(unique_langs):
                if i == j:
                    sdi_matrix[i, j] = 0
                    continue
                
                indices2 = [idx for idx, l in enumerate(labels) if l == lang2]
                if not indices2:
                    continue
                
                emb2 = embeddings[indices2]
                
                # Compute SDI = 1 - average cosine similarity
                sim_matrix = cosine_similarity(emb1, emb2)
                avg_sim = np.mean(sim_matrix)
                sdi_matrix[i, j] = 1 - avg_sim
        
        # Normalize SDI to [0, 1]
        sdi_matrix = np.clip(sdi_matrix, 0, 1)
        
        return pd.DataFrame(sdi_matrix, index=unique_langs, columns=unique_langs)
    
    def classify_sdi_scores(self, sdi_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Classify SDI scores into bias levels"""
        if sdi_matrix.empty:
            return {'average_sdi': 0, 'bias_level': 'Unknown', 'pair_classifications': {}}
        
        # Get all non-diagonal values
        values = []
        pair_classifications = {}
        
        for i, lang1 in enumerate(sdi_matrix.index):
            for j, lang2 in enumerate(sdi_matrix.columns):
                if i != j:
                    sdi = sdi_matrix.iloc[i, j]
                    values.append(sdi)
                    pair_classifications[f"{lang1} vs {lang2}"] = {
                        'sdi': sdi,
                        'bias_level': 'High' if sdi > SDI_THRESHOLD_HIGH else 'Moderate' if sdi > SDI_THRESHOLD_MODERATE else 'Low'
                    }
        
        avg_sdi = np.mean(values) if values else 0
        
        return {
            'average_sdi': avg_sdi,
            'bias_level': 'High' if avg_sdi > SDI_THRESHOLD_HIGH else 'Moderate' if avg_sdi > SDI_THRESHOLD_MODERATE else 'Low',
            'pair_classifications': pair_classifications
        }
    
    def perform_root_cause_analysis(self,
                                   embeddings: np.ndarray,
                                   labels: List[str],
                                   questions_by_lang: Dict[str, List[str]],
                                   answers_by_lang: Dict[str, List[str]],
                                   sdi_matrix: pd.DataFrame) -> List[Dict]:
        """Perform root cause analysis for high SDI pairs"""
        rca_results = []
        
        # Identify problematic pairs (high SDI)
        problematic_pairs = []
        for i, lang1 in enumerate(sdi_matrix.index):
            for j, lang2 in enumerate(sdi_matrix.columns):
                if i != j and sdi_matrix.iloc[i, j] > SDI_THRESHOLD_HIGH:
                    problematic_pairs.append((lang1, lang2, sdi_matrix.iloc[i, j]))
        
        # Sort by SDI severity
        problematic_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for lang1, lang2, sdi in problematic_pairs[:RCA_TOP_K]:
            # Find specific questions where divergence is highest
            if embeddings.size == 0 or lang1 not in answers_by_lang or lang2 not in answers_by_lang:
                continue
            
            # Get indices for this language pair
            indices1 = [idx for idx, l in enumerate(labels) if l == lang1]
            indices2 = [idx for idx, l in enumerate(labels) if l == lang2]
            
            if not indices1 or not indices2:
                continue
            
            # Compute pairwise similarities
            emb1 = embeddings[indices1]
            emb2 = embeddings[indices2]
            sim_matrix = cosine_similarity(emb1, emb2)
            
            # Find worst matching pairs
            worst_matches = []
            for i, idx1 in enumerate(indices1):
                # Get the lowest similarity for this English response
                min_sim = np.min(sim_matrix[i])
                min_idx = np.argmin(sim_matrix[i])
                
                worst_matches.append({
                    'lang1_idx': idx1,
                    'lang2_idx': indices2[min_idx],
                    'similarity': min_sim,
                    'english_text': answers_by_lang[lang1][idx1][:200] if idx1 < len(answers_by_lang[lang1]) else '',
                    'target_text': answers_by_lang[lang2][indices2[min_idx]][:200] if indices2[min_idx] < len(answers_by_lang[lang2]) else ''
                })
            
            # Sort by lowest similarity
            worst_matches.sort(key=lambda x: x['similarity'])
            top_matches = worst_matches[:5]
            
            # Determine root cause
            root_cause = self._determine_root_cause(top_matches, lang1, lang2)
            
            rca_results.append({
                'source_language': lang1,
                'target_language': lang2,
                'sdi_score': sdi,
                'severity': 'High' if sdi > 0.6 else 'Moderate',
                'root_cause': root_cause,
                'examples': [
                    {
                        'english_text': m['english_text'],
                        'target_text': m['target_text'],
                        'similarity': round(m['similarity'], 3)
                    } for m in top_matches
                ]
            })
        
        return rca_results
    
    def _determine_root_cause(self, examples: List[Dict], lang1: str, lang2: str) -> str:
        """Determine root cause from example mismatches"""
        if not examples:
            return "Unknown"
        
        # Check for length disparities
        avg_len_ratio = np.mean([len(e['english_text']) / max(len(e['target_text']), 1) for e in examples])
        if avg_len_ratio > 2.0:
            return "Translation Length Disparity"
        elif avg_len_ratio < 0.5:
            return "Content Omission"
        
        # Check for potential cultural differences (simplified)
        cultural_keywords = ['herb', 'traditional', 'clinic', 'doctor', 'hospital', 'family']
        
        cultural_mismatches = 0
        for e in examples:
            eng_has = any(kw in e['english_text'].lower() for kw in cultural_keywords)
            tgt_has = any(kw in e['target_text'].lower() for kw in cultural_keywords)
            if eng_has != tgt_has:
                cultural_mismatches += 1
        
        if cultural_mismatches > len(examples) / 2:
            return "Cultural Conceptual Mismatch"
        
        return "Semantic Divergence"
    
    def categorize_errors(self, rca_results: List[Dict]) -> Dict[str, int]:
        """Categorize errors from root cause analysis"""
        error_counts = defaultdict(int)
        
        for result in rca_results:
            error_counts[result['root_cause']] += 1
        
        return dict(error_counts)
    
    def compute_alignment_scores(self,
                                embeddings: np.ndarray,
                                labels: List[str]) -> pd.DataFrame:
        """Compute alignment scores between languages"""
        if embeddings.size == 0:
            return pd.DataFrame()
        
        unique_langs = list(set(labels))
        alignment_matrix = np.zeros((len(unique_langs), len(unique_langs)))
        
        for i, lang1 in enumerate(unique_langs):
            indices1 = [idx for idx, l in enumerate(labels) if l == lang1]
            if not indices1:
                continue
            
            emb1 = embeddings[indices1]
            centroid1 = np.mean(emb1, axis=0)
            
            for j, lang2 in enumerate(unique_langs):
                if i == j:
                    alignment_matrix[i, j] = 1.0
                    continue
                
                indices2 = [idx for idx, l in enumerate(labels) if l == lang2]
                if not indices2:
                    continue
                
                emb2 = embeddings[indices2]
                centroid2 = np.mean(emb2, axis=0)
                
                # Cosine similarity between centroids
                alignment_matrix[i, j] = np.dot(centroid1, centroid2) / (
                    np.linalg.norm(centroid1) * np.linalg.norm(centroid2) + 1e-8
                )
        
        alignment_matrix = np.clip(alignment_matrix, -1, 1)
        
        return pd.DataFrame(alignment_matrix, index=unique_langs, columns=unique_langs)
    
    def generate_cross_lingual_flags(self,
                                    sdi_classification: Dict,
                                    rca_results: List[Dict]) -> List[Dict]:
        """Generate flags from cross-lingual evaluation"""
        flags = []
        
        # High SDI flags
        if sdi_classification['bias_level'] == 'High':
            flags.append({
                'Type': 'High_SDI',
                'Severity': 'High',
                'Description': f"Average SDI of {sdi_classification['average_sdi']:.3f} indicates strong cross-lingual bias",
                'Recommendation': 'Review translation quality and cultural adaptation across all languages'
            })
        
        # Individual pair flags
        for pair, info in sdi_classification.get('pair_classifications', {}).items():
            if info['bias_level'] == 'High':
                flags.append({
                    'Type': 'Pairwise_Bias',
                    'Comparison': pair,
                    'Severity': 'High',
                    'Description': f"SDI of {info['sdi']:.3f} indicates high bias between {pair}",
                    'Recommendation': 'Focus translation quality improvement on this language pair'
                })
        
        # Root cause flags
        for result in rca_results:
            if result['severity'] == 'High':
                flags.append({
                    'Type': 'Root_Cause',
                    'Comparison': f"{result['source_language']} vs {result['target_language']}",
                    'Severity': 'High',
                    'Description': f"Root cause identified: {result['root_cause']} (SDI: {result['sdi_score']:.3f})",
                    'Recommendation': f"Address {result['root_cause'].lower()} in translation process"
                })
        
        return flags
    
    def run_full_evaluation(self,
                           embeddings: np.ndarray,
                           questions_by_lang: Dict[str, List[str]],
                           answers_by_lang: Dict[str, List[str]]) -> Dict:
        """Run complete cross-lingual evaluation"""
        self.logger.info("Starting cross-lingual evaluation")
        
        # Get labels from embeddings or create from answers
        if len(embeddings) > 0 and embeddings.shape[0] == sum(len(v) for v in answers_by_lang.values()):
            # Create labels matching embeddings
            labels = []
            for lang in PRIMARY_LANGUAGES:
                if lang in answers_by_lang:
                    labels.extend([lang] * len(answers_by_lang[lang]))
        else:
            # Fallback: create dummy labels
            labels = []
            for lang in PRIMARY_LANGUAGES:
                if lang in answers_by_lang:
                    labels.extend([lang] * len(answers_by_lang[lang]))
        
        # Compute SDI matrix
        sdi_matrix = self.compute_semantic_distance_index(embeddings, labels)
        
        # Classify SDI scores
        sdi_classification = self.classify_sdi_scores(sdi_matrix)
        
        # Compute alignment scores
        alignment_scores = self.compute_alignment_scores(embeddings, labels)
        
        # Perform root cause analysis
        rca_results = self.perform_root_cause_analysis(
            embeddings, labels, questions_by_lang, answers_by_lang, sdi_matrix
        )
        
        # Categorize errors
        error_categories = self.categorize_errors(rca_results)
        
        # Generate flags
        flags = self.generate_cross_lingual_flags(sdi_classification, rca_results)
        
        # Summary
        summary = {
            'languages_evaluated': list(sdi_matrix.index) if not sdi_matrix.empty else [],
            'average_sdi': sdi_classification['average_sdi'],
            'bias_level': sdi_classification['bias_level'],
            'high_sdi_pairs': len([p for p, i in sdi_classification.get('pair_classifications', {}).items() if i['bias_level'] == 'High']),
            'root_causes_identified': len(rca_results),
            'flags_generated': len(flags)
        }
        
        results = {
            'sdi_matrix': sdi_matrix,
            'sdi_classification': sdi_classification,
            'alignment_scores': alignment_scores,
            'rca_results': rca_results,
            'error_categories': error_categories,
            'flags': flags,
            'summary': summary
        }
        
        self.logger.info(f"Cross-lingual evaluation complete: {summary}")
        return results