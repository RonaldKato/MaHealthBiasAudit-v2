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
        if embeddings is None or embeddings.size == 0:
            self.logger.warning("No embeddings available for SDI computation")
            return pd.DataFrame(columns=['Unknown'], index=['Unknown'])
        
        # Check if embeddings are valid
        if np.all(embeddings == 0) or np.std(embeddings) < 1e-6:
            self.logger.warning("Embeddings are all zeros or have no variance")
            return pd.DataFrame(columns=['Unknown'], index=['Unknown'])
        
        unique_langs = list(set(labels))
        if len(unique_langs) < 2:
            self.logger.warning(f"Only {len(unique_langs)} language found, cannot compute SDI")
            return pd.DataFrame(index=unique_langs, columns=unique_langs)
        
        self.logger.info(f"Computing SDI for {len(unique_langs)} languages with {len(embeddings)} samples")
        
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
                
                # Compute cosine similarity
                try:
                    sim_matrix = cosine_similarity(emb1, emb2)
                    avg_sim = np.mean(sim_matrix)
                    
                    # Sanity check: if avg_sim is nan or outside [-1,1], fix it
                    if np.isnan(avg_sim):
                        avg_sim = 0.0
                    avg_sim = np.clip(avg_sim, -1, 1)
                    
                    sdi_matrix[i, j] = 1 - avg_sim
                    
                    # Sanity check: ensure SDI is in [0, 2] range
                    sdi_matrix[i, j] = np.clip(sdi_matrix[i, j], 0, 2)
                    
                except Exception as e:
                    self.logger.warning(f"Error computing similarity between {lang1} and {lang2}: {e}")
                    sdi_matrix[i, j] = 0.5  # Default moderate difference
        
        # Normalize SDI values to [0, 1] range for better interpretation
        # But preserve relative differences
        sdi_matrix = np.clip(sdi_matrix, 0, 1)
        
        df = pd.DataFrame(sdi_matrix, index=unique_langs, columns=unique_langs)
        
        # Log the SDI matrix for debugging
        self.logger.info(f"SDI matrix:\n{df}")
        
        return df

    def classify_sdi_scores(self, sdi_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Classify SDI scores into bias levels"""
        if sdi_matrix is None or sdi_matrix.empty:
            return {
                'average_sdi': 0.0,
                'bias_level': 'Unknown',
                'percentage': '0.0%',
                'pair_classifications': {},
                'total_pairs': 0,
                'high_pairs': 0,
                'moderate_pairs': 0,
                'low_pairs': 0
            }
        
        values = []
        pair_classifications = {}
        
        for i, lang1 in enumerate(sdi_matrix.index):
            for j, lang2 in enumerate(sdi_matrix.columns):
                if i != j:
                    sdi = sdi_matrix.iloc[i, j]
                    values.append(sdi)
                    
                    if sdi > SDI_THRESHOLD_HIGH:
                        bias_level = 'High'
                    elif sdi > SDI_THRESHOLD_MODERATE:
                        bias_level = 'Moderate'
                    else:
                        bias_level = 'Low'
                    
                    pair_classifications[f"{lang1} vs {lang2}"] = {
                        'sdi': float(sdi),
                        'bias_level': bias_level,
                        'percentage': f"{sdi*100:.1f}%"
                    }
        
        avg_sdi = np.mean(values) if values else 0.0
        
        # Use consistent thresholds
        if avg_sdi > 0.4:
            bias_level = 'HIGH'
        elif avg_sdi > 0.2:
            bias_level = 'MODERATE'
        else:
            bias_level = 'LOW'
        
        return {
            'average_sdi': float(avg_sdi),
            'bias_level': bias_level,
            'percentage': f"{avg_sdi*100:.1f}%",
            'pair_classifications': pair_classifications,
            'total_pairs': len(pair_classifications),
            'high_pairs': sum(1 for p in pair_classifications.values() if p['bias_level'] == 'High'),
            'moderate_pairs': sum(1 for p in pair_classifications.values() if p['bias_level'] == 'Moderate'),
            'low_pairs': sum(1 for p in pair_classifications.values() if p['bias_level'] == 'Low')
        }
    
    def perform_root_cause_analysis(self,
                                   embeddings: np.ndarray,
                                   labels: List[str],
                                   questions_by_lang: Dict[str, List[str]],
                                   answers_by_lang: Dict[str, List[str]],
                                   sdi_matrix: pd.DataFrame) -> List[Dict]:
        """Perform root cause analysis for high SDI pairs"""
        rca_results = []
        
        problematic_pairs = []
        for i, lang1 in enumerate(sdi_matrix.index):
            for j, lang2 in enumerate(sdi_matrix.columns):
                if i != j and sdi_matrix.iloc[i, j] > SDI_THRESHOLD_HIGH:
                    problematic_pairs.append((lang1, lang2, sdi_matrix.iloc[i, j]))
        
        problematic_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for lang1, lang2, sdi in problematic_pairs[:RCA_TOP_K]:
            if embeddings.size == 0 or lang1 not in answers_by_lang or lang2 not in answers_by_lang:
                continue
            
            indices1 = [idx for idx, l in enumerate(labels) if l == lang1]
            indices2 = [idx for idx, l in enumerate(labels) if l == lang2]
            
            if not indices1 or not indices2:
                continue
            
            emb1 = embeddings[indices1]
            emb2 = embeddings[indices2]
            sim_matrix = cosine_similarity(emb1, emb2)
            
            worst_matches = []
            for i, idx1 in enumerate(indices1):
                min_sim = np.min(sim_matrix[i])
                min_idx = np.argmin(sim_matrix[i])
                
                source_text = answers_by_lang[lang1][idx1] if idx1 < len(answers_by_lang[lang1]) else ''
                target_text = answers_by_lang[lang2][indices2[min_idx]] if indices2[min_idx] < len(answers_by_lang[lang2]) else ''
                
                worst_matches.append({
                    'lang1_idx': idx1,
                    'lang2_idx': indices2[min_idx],
                    'similarity': float(min_sim),
                    'source_text': source_text[:300] + '...' if len(source_text) > 300 else source_text,
                    'target_text': target_text[:300] + '...' if len(target_text) > 300 else target_text
                })
            
            worst_matches.sort(key=lambda x: x['similarity'])
            top_matches = worst_matches[:5]
            
            root_cause, detailed_reason = self._determine_root_cause_detailed(top_matches, lang1, lang2)
            
            rca_results.append({
                'source_language': lang1,
                'target_language': lang2,
                'sdi_score': float(sdi),
                'sdi_percentage': f"{sdi*100:.1f}%",
                'severity': 'Critical' if sdi > 0.6 else 'High',
                'root_cause': root_cause,
                'detailed_reason': detailed_reason,
                'examples': [
                    {
                        'source_text': m['source_text'],
                        'target_text': m['target_text'],
                        'similarity': m['similarity'],
                        'similarity_percentage': f"{m['similarity']*100:.1f}%"
                    } for m in top_matches
                ]
            })
        
        return rca_results
    
    def _determine_root_cause_detailed(self, examples: List[Dict], lang1: str, lang2: str) -> Tuple[str, str]:
        """Determine root cause with detailed reason"""
        if not examples:
            return "Unknown", "No examples available"
        
        # Check for length disparity
        avg_len_ratio = np.mean([len(e['source_text']) / max(len(e['target_text']), 1) for e in examples])
        
        if avg_len_ratio > 3.0:
            return "Severe Length Disparity", f"Target language responses are only {1/avg_len_ratio:.1%} of source length"
        elif avg_len_ratio > 2.0:
            return "Translation Length Disparity", f"Target language responses are {1/avg_len_ratio:.1%} of source length"
        elif avg_len_ratio < 0.4:
            return "Severe Content Omission", f"Target language responses are {avg_len_ratio:.1%} of source length"
        elif avg_len_ratio < 0.6:
            return "Content Omission", f"Target language responses are {avg_len_ratio:.1%} of source length"
        
        # Check for cultural keywords
        cultural_keywords = ['herb', 'traditional', 'clinic', 'doctor', 'hospital', 'family', 'mitishamba', 'ddagala', 'omuddo']
        
        cultural_mismatches = 0
        for e in examples:
            src_has = any(kw in e['source_text'].lower() for kw in cultural_keywords)
            tgt_has = any(kw in e['target_text'].lower() for kw in cultural_keywords)
            if src_has != tgt_has:
                cultural_mismatches += 1
        
        if cultural_mismatches > len(examples) / 2:
            return "Cultural Conceptual Mismatch", f"{cultural_mismatches}/{len(examples)} examples show cultural content divergence"
        
        # Check for medical term omission
        medical_keywords = ['preeclampsia', 'hypertension', 'diabetes', 'anemia', 'infection', 'medication', 'vaccination']
        
        medical_mismatches = 0
        for e in examples:
            src_has = any(kw in e['source_text'].lower() for kw in medical_keywords)
            tgt_has = any(kw in e['target_text'].lower() for kw in medical_keywords)
            if src_has and not tgt_has:
                medical_mismatches += 1
        
        if medical_mismatches > len(examples) / 3:
            return "Medical Term Omission", f"{medical_mismatches}/{len(examples)} examples show missing medical terminology"
        
        # Check for negation
        negation_keywords = ['not', 'no', 'never', 'don\'t', 'cannot', 'si', 'hapana', 'ta', 'sili']
        
        negation_mismatches = 0
        for e in examples:
            src_has = any(kw in e['source_text'].lower() for kw in negation_keywords)
            tgt_has = any(kw in e['target_text'].lower() for kw in negation_keywords)
            if src_has != tgt_has:
                negation_mismatches += 1
        
        if negation_mismatches > len(examples) / 3:
            return "Negation Misinterpretation", f"{negation_mismatches}/{len(examples)} examples show negation issues"
        
        # Check for emotional tone
        emotion_keywords = ['urgent', 'immediate', 'serious', 'important', 'please', 'careful', 'emergency']
        
        emotion_mismatches = 0
        for e in examples:
            src_has = any(kw in e['source_text'].lower() for kw in emotion_keywords)
            tgt_has = any(kw in e['target_text'].lower() for kw in emotion_keywords)
            if src_has and not tgt_has:
                emotion_mismatches += 1
        
        if emotion_mismatches > len(examples) / 3:
            return "Emotional Tone Shift", f"{emotion_mismatches}/{len(examples)} examples show loss of urgency/empathy"
        
        return "Semantic Divergence", "General semantic differences without specific pattern"
    
    def categorize_errors(self, rca_results: List[Dict]) -> Dict[str, int]:
        """Categorize errors from root cause analysis"""
        if not rca_results:
            return {
                'by_type': {},
                'by_severity': {},
                'total': 0
            }
        
        error_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for result in rca_results:
            error_counts[result.get('root_cause', 'Unknown')] += 1
            severity_counts[result.get('severity', 'Unknown')] += 1
        
        return {
            'by_type': dict(error_counts),
            'by_severity': dict(severity_counts),
            'total': len(rca_results)
        }
    
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
                
                alignment_matrix[i, j] = np.dot(centroid1, centroid2) / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2) + 1e-8)
        
        alignment_matrix = np.clip(alignment_matrix, -1, 1)
        
        return pd.DataFrame(alignment_matrix, index=unique_langs, columns=unique_langs)
    
    def generate_cross_lingual_flags(self,
                                    sdi_classification: Dict,
                                    rca_results: List[Dict],
                                    answers_by_lang: Dict) -> List[Dict]:
        """Generate flags from cross-lingual evaluation with enhanced severity"""
        flags = []
        
        # Overall bias level
        if sdi_classification['bias_level'] == 'High':
            flags.append({
                'Type': 'High_SDI_Critical',
                'Severity': 'Critical',
                'Description': f"Average SDI of {sdi_classification['average_sdi']:.3f} ({sdi_classification['percentage']}) indicates strong cross-lingual bias",
                'Recommendation': 'URGENT: Review translation quality and cultural adaptation across all languages'
            })
        elif sdi_classification['bias_level'] == 'Moderate':
            flags.append({
                'Type': 'High_SDI',
                'Severity': 'High',
                'Description': f"Average SDI of {sdi_classification['average_sdi']:.3f} ({sdi_classification['percentage']}) indicates moderate cross-lingual bias",
                'Recommendation': 'Review translation quality and cultural adaptation across all languages'
            })
        
        # Pairwise biases
        for pair, info in sdi_classification.get('pair_classifications', {}).items():
            if info['bias_level'] == 'High':
                flags.append({
                    'Type': 'Pairwise_Bias_Critical',
                    'Comparison': pair,
                    'Severity': 'Critical' if info['sdi'] > 0.6 else 'High',
                    'Description': f"SDI of {info['sdi']:.3f} ({info['percentage']}) indicates high bias between {pair}",
                    'Recommendation': 'URGENT: Focus translation quality improvement on this language pair'
                })
            elif info['bias_level'] == 'Moderate':
                flags.append({
                    'Type': 'Pairwise_Bias',
                    'Comparison': pair,
                    'Severity': 'Moderate',
                    'Description': f"SDI of {info['sdi']:.3f} ({info['percentage']}) indicates moderate bias between {pair}",
                    'Recommendation': 'Review translation quality for this language pair'
                })
        
        # Root cause flags
        for result in rca_results:
            if result['severity'] == 'Critical':
                flags.append({
                    'Type': 'Root_Cause_Critical',
                    'Comparison': f"{result['source_language']} vs {result['target_language']}",
                    'Severity': 'Critical',
                    'Description': f"Critical root cause: {result['root_cause']} (SDI: {result['sdi_percentage']})",
                    'Recommendation': f"URGENT: Address {result['root_cause'].lower()} in translation process",
                    'Root_Cause': result['root_cause']
                })
            elif result['severity'] == 'High':
                flags.append({
                    'Type': 'Root_Cause',
                    'Comparison': f"{result['source_language']} vs {result['target_language']}",
                    'Severity': 'High',
                    'Description': f"Root cause: {result['root_cause']} (SDI: {result['sdi_percentage']})",
                    'Recommendation': f"Address {result['root_cause'].lower()} in translation process",
                    'Root_Cause': result['root_cause']
                })
        
        return flags
    
    def run_full_evaluation(self,
                        embeddings: np.ndarray,
                        questions_by_lang: Dict[str, List[str]],
                        answers_by_lang: Dict[str, List[str]]) -> Dict:
        """Run complete cross-lingual evaluation"""
        self.logger.info("="*50)
        self.logger.info("STARTING CROSS-LINGUAL EVALUATION")
        self.logger.info("="*50)
        
        # Check if we have data
        if embeddings is None or embeddings.size == 0:
            self.logger.warning("No embeddings available for cross-lingual evaluation")
            languages = list(answers_by_lang.keys()) if answers_by_lang else ['Unknown']
            return {
                'sdi_matrix': pd.DataFrame(index=languages, columns=languages),
                'sdi_classification': {
                    'average_sdi': 0.0,
                    'bias_level': 'Unknown',
                    'percentage': '0.0%',
                    'pair_classifications': {}
                },
                'alignment_scores': pd.DataFrame(),
                'rca_results': [],
                'error_categories': {'by_type': {}, 'by_severity': {}, 'total': 0},
                'flags': [],
                'summary': {
                    'languages_evaluated': [],
                    'average_sdi': 0.0,
                    'average_sdi_percentage': '0.0%',
                    'bias_level': 'Unknown',
                    'high_sdi_pairs': 0,
                    'moderate_sdi_pairs': 0,
                    'low_sdi_pairs': 0,
                    'root_causes_identified': 0,
                    'critical_root_causes': 0,
                    'flags_generated': 0,
                    'critical_flags': 0
                }
            }
        
        # Log embedding stats
        self.logger.info(f"Embeddings shape: {embeddings.shape}")
        self.logger.info(f"Embeddings mean: {np.mean(embeddings):.6f}")
        self.logger.info(f"Embeddings std: {np.std(embeddings):.6f}")
        self.logger.info(f"Embeddings min: {np.min(embeddings):.6f}")
        self.logger.info(f"Embeddings max: {np.max(embeddings):.6f}")
        
        # Check if embeddings are valid
        if np.all(embeddings == 0):
            self.logger.warning("All embeddings are zero!")
            languages = list(answers_by_lang.keys()) if answers_by_lang else ['Unknown']
            return {
                'sdi_matrix': pd.DataFrame(index=languages, columns=languages),
                'sdi_classification': {
                    'average_sdi': 0.0,
                    'bias_level': 'Unknown',
                    'percentage': '0.0%',
                    'pair_classifications': {}
                },
                'alignment_scores': pd.DataFrame(),
                'rca_results': [],
                'error_categories': {'by_type': {}, 'by_severity': {}, 'total': 0},
                'flags': [],
                'summary': {
                    'languages_evaluated': [],
                    'average_sdi': 0.0,
                    'average_sdi_percentage': '0.0%',
                    'bias_level': 'Unknown',
                    'high_sdi_pairs': 0,
                    'moderate_sdi_pairs': 0,
                    'low_sdi_pairs': 0,
                    'root_causes_identified': 0,
                    'critical_root_causes': 0,
                    'flags_generated': 0,
                    'critical_flags': 0
                }
            }
        
        # Build labels from answers
        labels = []
        for lang in PRIMARY_LANGUAGES:
            if lang in answers_by_lang and answers_by_lang[lang]:
                labels.extend([lang] * len(answers_by_lang[lang]))
        
        if not labels:
            self.logger.warning("No labels available for cross-lingual evaluation")
            languages = list(answers_by_lang.keys()) if answers_by_lang else ['Unknown']
            return {
                'sdi_matrix': pd.DataFrame(index=languages, columns=languages),
                'sdi_classification': {
                    'average_sdi': 0.0,
                    'bias_level': 'Unknown',
                    'percentage': '0.0%',
                    'pair_classifications': {}
                },
                'alignment_scores': pd.DataFrame(),
                'rca_results': [],
                'error_categories': {'by_type': {}, 'by_severity': {}, 'total': 0},
                'flags': [],
                'summary': {
                    'languages_evaluated': [],
                    'average_sdi': 0.0,
                    'average_sdi_percentage': '0.0%',
                    'bias_level': 'Unknown',
                    'high_sdi_pairs': 0,
                    'moderate_sdi_pairs': 0,
                    'low_sdi_pairs': 0,
                    'root_causes_identified': 0,
                    'critical_root_causes': 0,
                    'flags_generated': 0,
                    'critical_flags': 0
                }
            }
        
        # Ensure embeddings match labels
        if len(embeddings) != len(labels):
            self.logger.warning(f"Embeddings count ({len(embeddings)}) doesn't match labels ({len(labels)})")
            min_len = min(len(embeddings), len(labels))
            embeddings = embeddings[:min_len]
            labels = labels[:min_len]
        
        # Log language distribution
        lang_counts = {lang: labels.count(lang) for lang in set(labels)}
        self.logger.info(f"Language distribution: {lang_counts}")
        
        self.logger.info(f"Computing SDI matrix for {len(set(labels))} languages...")
        sdi_matrix = self.compute_semantic_distance_index(embeddings, labels)
        
        # Log the SDI matrix for debugging
        if not sdi_matrix.empty:
            self.logger.info(f"SDI matrix values:\n{sdi_matrix}")
        
        self.logger.info("Classifying SDI scores...")
        sdi_classification = self.classify_sdi_scores(sdi_matrix)
        self.logger.info(f"SDI Classification: {sdi_classification}")
        
        # Check if SDI values are reasonable
        avg_sdi = sdi_classification.get('average_sdi', 0)
        if avg_sdi > 0.8:
            self.logger.warning(f"Very high SDI ({avg_sdi:.4f}) - embeddings may be problematic")
            self.logger.warning("Using TF-IDF fallback for more meaningful embeddings...")
            
            # Try to use TF-IDF fallback embeddings
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Collect all texts
                all_texts = []
                for lang, texts in answers_by_lang.items():
                    all_texts.extend(texts)
                
                if all_texts:
                    vectorizer = TfidfVectorizer(max_features=300, stop_words=None)
                    tfidf_embeddings = vectorizer.fit_transform(all_texts).toarray()
                    
                    # Normalize
                    row_norms = np.linalg.norm(tfidf_embeddings, axis=1, keepdims=True)
                    tfidf_embeddings = tfidf_embeddings / (row_norms + 1e-8)
                    
                    self.logger.info(f"TF-IDF embeddings shape: {tfidf_embeddings.shape}")
                    
                    # Recompute SDI with TF-IDF
                    sdi_matrix = self.compute_semantic_distance_index(tfidf_embeddings, labels)
                    sdi_classification = self.classify_sdi_scores(sdi_matrix)
                    self.logger.info(f"TF-IDF SDI Classification: {sdi_classification}")
            except Exception as e:
                self.logger.warning(f"TF-IDF fallback failed: {e}")
        
        self.logger.info("Computing alignment scores...")
        alignment_scores = self.compute_alignment_scores(embeddings, labels)
        
        self.logger.info("Performing root cause analysis...")
        rca_results = self.perform_root_cause_analysis(embeddings, labels, questions_by_lang, answers_by_lang, sdi_matrix)
        
        self.logger.info("Categorizing errors...")
        error_categories = self.categorize_errors(rca_results)
        
        self.logger.info("Generating flags...")
        flags = self.generate_cross_lingual_flags(sdi_classification, rca_results, answers_by_lang)
        
        summary = {
            'languages_evaluated': list(sdi_matrix.index) if not sdi_matrix.empty else [],
            'average_sdi': sdi_classification.get('average_sdi', 0),
            'average_sdi_percentage': sdi_classification.get('percentage', 'N/A'),
            'bias_level': sdi_classification.get('bias_level', 'Unknown'),
            'high_sdi_pairs': sdi_classification.get('high_pairs', 0),
            'moderate_sdi_pairs': sdi_classification.get('moderate_pairs', 0),
            'low_sdi_pairs': sdi_classification.get('low_pairs', 0),
            'root_causes_identified': len(rca_results) if rca_results else 0,
            'critical_root_causes': sum(1 for r in (rca_results or []) if r.get('severity') == 'Critical'),
            'flags_generated': len(flags) if flags else 0,
            'critical_flags': sum(1 for f in (flags or []) if f.get('Severity') == 'Critical')
        }
        
        results = {
            'sdi_matrix': sdi_matrix,
            'sdi_classification': sdi_classification,
            'alignment_scores': alignment_scores,
            'rca_results': rca_results or [],
            'error_categories': error_categories or {'by_type': {}, 'by_severity': {}, 'total': 0},
            'flags': flags or [],
            'summary': summary
        }
        
        self.logger.info("="*50)
        self.logger.info("CROSS-LINGUAL EVALUATION COMPLETE")
        self.logger.info(f"  Languages: {summary['languages_evaluated']}")
        self.logger.info(f"  Average SDI: {summary['average_sdi']:.3f} ({summary['average_sdi_percentage']})")
        self.logger.info(f"  Bias Level: {summary['bias_level']}")
        self.logger.info("="*50)
        
        return results
    
    def run_experiments(self) -> List[Dict]:
        """Run experiments with increasing dataset sizes (10, 100, 1000, 10000)"""
        print("\n" + "="*70)
        print(" RUNNING EXPERIMENTS")
        print(f"   Sample sizes: {EXPERIMENT_SIZES} (increasing by squares)")
        print("="*70)
        
        data_dict = self.load_maternal_multilingual_dataset()
        
        # Log original data size
        original_counts = {}
        for cat, cat_data in data_dict.items():
            for lang, answers in cat_data.get('answers', {}).items():
                original_counts[lang] = original_counts.get(lang, 0) + len(answers)
        print(f"Original dataset sizes: {original_counts}")
        
        experiment_reports = []
        
        for size in EXPERIMENT_SIZES:
            print(f"\n Experiment: Sample Size = {size}")
            print(f"   {'─'*40}")
            
            # Sample the dataset
            sampled_data = self._sample_dataset(data_dict, size)
            
            # Log sampled data size
            sampled_counts = {}
            for cat, cat_data in sampled_data.items():
                for lang, answers in cat_data.get('answers', {}).items():
                    sampled_counts[lang] = sampled_counts.get(lang, 0) + len(answers)
            print(f"   Sampled sizes: {sampled_counts}")
            
            if sum(sampled_counts.values()) == 0:
                print(f" No data sampled for size {size}")
                experiment_reports.append({'sample_size': size, 'error': 'No data sampled'})
                continue
            
            # Track metrics
            start_time = time.time()
            
            try:
                report = self.run_pipeline(sampled_data, f"experiment_{size}")
                execution_time = time.time() - start_time
                
                experiment_reports.append({
                    'sample_size': size,
                    'avg_sdi': report['key_metrics']['average_sdi'],
                    'sdi_percentage': report['key_metrics'].get('sdi_percentage', 'N/A'),
                    'bias_level': report['key_metrics']['bias_level'],
                    'total_flags': report['key_metrics']['total_flags'],
                    'critical_flags': report['key_metrics'].get('critical_flags', 0),
                    'execution_time': round(execution_time, 2)
                })
                
                print(f"  Completed in {execution_time:.2f}s")
                print(f"  SDI: {report['key_metrics']['average_sdi']:.4f}")
                print(f"  Flags: {report['key_metrics']['total_flags']}")
                
            except Exception as e:
                print(f"  Failed: {e}")
                import traceback
                traceback.print_exc()
                experiment_reports.append({
                    'sample_size': size, 
                    'error': str(e)
                })
        
        self._save_experiment_results(experiment_reports)
        self._plot_experiment_performance(experiment_reports)
        
        return experiment_reports