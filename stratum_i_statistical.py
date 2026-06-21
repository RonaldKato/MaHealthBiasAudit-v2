"""
MaHealthBiasAudit - Stratum I: Statistical Bias Audit
Statistical analysis of response characteristics across languages
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import PRIMARY_LANGUAGES, SIGNIFICANCE_ALPHA
from utils import setup_logger, basic_tokenize, compute_vocabulary_richness, compute_lexical_diversity, cohens_d


class StatisticalBiasAuditor:
    """Statistical analysis of bias across languages"""
    
    def __init__(self):
        self.logger = setup_logger('statistical')
        self.results = {}
    
    def compute_response_length_stats(self, 
                                      normalized_texts: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute response length statistics per language"""
        stats = []
        
        for lang, texts in normalized_texts.items():
            if not texts:
                continue
            lengths = [len(t.split()) for t in texts]
            
            stats.append({
                'Language': lang,
                'Count': len(texts),
                'Mean': np.mean(lengths),
                'Std': np.std(lengths),
                'Min': np.min(lengths),
                '25%': np.percentile(lengths, 25),
                '50%': np.percentile(lengths, 50),
                '75%': np.percentile(lengths, 75),
                'Max': np.max(lengths)
            })
        
        return pd.DataFrame(stats)
    
    def compute_vocabulary_richness_stats(self, 
                                          normalized_texts: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute vocabulary richness metrics per language"""
        stats = []
        
        for lang, texts in normalized_texts.items():
            if not texts:
                continue
            richness_scores = []
            diversity_scores = []
            
            for text in texts:
                tokens = basic_tokenize(text)
                if tokens:
                    richness_scores.append(compute_vocabulary_richness(tokens))
                    diversity_scores.append(compute_lexical_diversity(tokens))
            
            stats.append({
                'Language': lang,
                'Vocabulary_Richness_Mean': np.mean(richness_scores) if richness_scores else 0,
                'Vocabulary_Richness_Std': np.std(richness_scores) if richness_scores else 0,
                'Lexical_Diversity_Mean': np.mean(diversity_scores) if diversity_scores else 0,
                'Lexical_Diversity_Std': np.std(diversity_scores) if diversity_scores else 0
            })
        
        return pd.DataFrame(stats)
    
    def compute_ngram_analysis(self, 
                               normalized_texts: Dict[str, List[str]], 
                               n: int = 2) -> pd.DataFrame:
        """Compute n-gram frequency analysis"""
        results = []
        
        for lang, texts in normalized_texts.items():
            if not texts:
                continue
            
            try:
                vectorizer = CountVectorizer(ngram_range=(n, n), lowercase=True, 
                                             tokenizer=basic_tokenize, token_pattern=None)
                ngram_matrix = vectorizer.fit_transform(texts)
                ngram_counts = np.array(ngram_matrix.sum(axis=0)).flatten()
                ngram_names = vectorizer.get_feature_names_out()
                
                top_indices = np.argsort(ngram_counts)[-20:][::-1]
                top_ngrams = [(ngram_names[i], int(ngram_counts[i])) for i in top_indices]
                
                results.append({
                    'Language': lang,
                    f'Top_{n}-grams': top_ngrams[:10],
                    'Unique_ngrams': len(ngram_names),
                    'Total_ngram_occurrences': int(sum(ngram_counts))
                })
            except Exception as e:
                self.logger.warning(f"Error computing {n}-grams for {lang}: {e}")
                results.append({'Language': lang, f'Top_{n}-grams': [], 'Unique_ngrams': 0, 'Total_ngram_occurrences': 0})
        
        return pd.DataFrame(results)
    
    def perform_statistical_tests(self, 
                                  normalized_texts: Dict[str, List[str]]) -> List[Dict]:
        """Perform statistical tests comparing languages"""
        results = []
        
        lang_data = {}
        for lang, texts in normalized_texts.items():
            if texts:
                lengths = [len(t.split()) for t in texts]
                lang_data[lang] = lengths
        
        lang_list = list(lang_data.keys())
        for i, lang1 in enumerate(lang_list):
            for lang2 in lang_list[i+1:]:
                data1 = lang_data[lang1]
                data2 = lang_data[lang2]
                
                try:
                    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                except:
                    u_stat, p_value = 0, 1.0
                
                d = cohens_d(np.array(data1), np.array(data2))
                
                if p_value < SIGNIFICANCE_ALPHA:
                    if abs(d) > 0.5:
                        interpretation = f"Large effect: {lang1} responses are significantly {'longer' if np.mean(data1) > np.mean(data2) else 'shorter'}"
                    elif abs(d) > 0.2:
                        interpretation = f"Medium effect: {lang1} responses are significantly {'longer' if np.mean(data1) > np.mean(data2) else 'shorter'}"
                    else:
                        interpretation = f"Small effect: {lang1} responses are significantly {'longer' if np.mean(data1) > np.mean(data2) else 'shorter'}"
                else:
                    interpretation = "No significant difference"
                
                results.append({
                    'Comparison': f"{lang1} vs {lang2}",
                    'Mean_Diff': round(np.mean(data1) - np.mean(data2), 2),
                    'Mann_Whitney_U': round(u_stat, 2),
                    'P_Value': round(p_value, 4),
                    'Significant': p_value < SIGNIFICANCE_ALPHA,
                    'Effect_Size_Cohens_d': round(d, 3),
                    'Interpretation': interpretation,
                    'Lang1_Mean': round(np.mean(data1), 2),
                    'Lang2_Mean': round(np.mean(data2), 2)
                })
        
        return results
    
    def detect_outliers(self, normalized_texts: Dict[str, List[str]]) -> List[Dict]:
        """Detect outlier responses that may indicate bias"""
        outliers = []
        
        for lang, texts in normalized_texts.items():
            if not texts:
                continue
            lengths = [len(t.split()) for t in texts]
            
            q1 = np.percentile(lengths, 25)
            q3 = np.percentile(lengths, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            for idx, length in enumerate(lengths):
                if length < lower_bound or length > upper_bound:
                    outliers.append({
                        'Language': lang,
                        'Index': idx,
                        'Length': length,
                        'Text_Preview': texts[idx][:150] + '...' if len(texts[idx]) > 150 else texts[idx],
                        'Outlier_Type': 'Short' if length < lower_bound else 'Long'
                    })
        
        return outliers
    
    def generate_flags(self, 
                       length_stats: pd.DataFrame,
                       vocab_stats: pd.DataFrame,
                       test_results: List[Dict]) -> List[Dict]:
        """Generate flags for potential biases with enhanced categorization"""
        flags = []
        
        if length_stats.empty:
            return flags
        
        # Get English as reference
        english_row = None
        for idx, row in length_stats.iterrows():
            if row.get('Language') == 'English':
                english_row = row
                break
        
        if english_row is not None:
            english_mean = english_row.get('Mean', 0)
            
            for idx, row in length_stats.iterrows():
                lang = row.get('Language')
                if lang != 'English' and lang:
                    mean_val = row.get('Mean', 0)
                    if mean_val > 0 and english_mean > 0:
                        ratio = mean_val / english_mean
                        if ratio < 0.5:
                            flags.append({
                                'Type': 'Length_Bias_Critical',
                                'Language': lang,
                                'Severity': 'Critical',
                                'Description': f"Responses are {ratio:.1%} of English length ({mean_val:.0f} vs {english_mean:.0f} words)",
                                'Recommendation': 'URGENT: Review translation quality and cultural adaptation',
                                'Ratio': round(ratio, 2)
                            })
                        elif ratio < 0.7:
                            flags.append({
                                'Type': 'Length_Bias_High',
                                'Language': lang,
                                'Severity': 'High',
                                'Description': f"Responses are {ratio:.1%} of English length ({mean_val:.0f} vs {english_mean:.0f} words)",
                                'Recommendation': 'Review if content loss is occurring',
                                'Ratio': round(ratio, 2)
                            })
                        elif ratio < 0.85:
                            flags.append({
                                'Type': 'Length_Bias_Moderate',
                                'Language': lang,
                                'Severity': 'Moderate',
                                'Description': f"Responses are {ratio:.1%} of English length",
                                'Recommendation': 'Monitor for potential content gaps',
                                'Ratio': round(ratio, 2)
                            })
        
        # Vocabulary richness flags
        if not vocab_stats.empty:
            for idx, row in vocab_stats.iterrows():
                lang = row.get('Language')
                if lang != 'English' and lang:
                    richness = row.get('Vocabulary_Richness_Mean', 0)
                    if richness < 0.25 and richness > 0:
                        flags.append({
                            'Type': 'Vocabulary_Bias_Critical',
                            'Language': lang,
                            'Severity': 'Critical',
                            'Description': f"Very low vocabulary richness ({richness:.3f})",
                            'Recommendation': 'URGENT: Check if data collection captures natural language variation',
                            'Value': round(richness, 3)
                        })
                    elif richness < 0.35:
                        flags.append({
                            'Type': 'Vocabulary_Bias_High',
                            'Language': lang,
                            'Severity': 'High',
                            'Description': f"Low vocabulary richness ({richness:.3f})",
                            'Recommendation': 'Review data collection and translation quality',
                            'Value': round(richness, 3)
                        })
        
        # Statistical significance flags
        for test in test_results:
            if test.get('Significant', False):
                effect = abs(test.get('Effect_Size_Cohens_d', 0))
                if effect > 0.7:
                    severity = 'Critical'
                elif effect > 0.5:
                    severity = 'High'
                elif effect > 0.3:
                    severity = 'Moderate'
                else:
                    severity = 'Low'
                
                flags.append({
                    'Type': 'Statistical_Bias',
                    'Comparison': test['Comparison'],
                    'Severity': severity,
                    'Description': test['Interpretation'],
                    'Recommendation': 'Investigate potential systematic differences',
                    'Effect_Size': round(effect, 3)
                })
        
        return flags
    
    def run_full_audit(self, 
                       questions_by_lang: Dict[str, List[str]], 
                       answers_by_lang: Dict[str, List[str]]) -> Dict:
        """Run complete statistical bias audit"""
        self.logger.info("="*50)
        self.logger.info("STARTING STATISTICAL BIAS AUDIT")
        self.logger.info("="*50)
        
        self.logger.info("Computing response length statistics...")
        length_stats = self.compute_response_length_stats(answers_by_lang)
        
        self.logger.info("Computing vocabulary richness...")
        vocab_stats = self.compute_vocabulary_richness_stats(answers_by_lang)
        
        self.logger.info("Performing statistical tests...")
        test_results = self.perform_statistical_tests(answers_by_lang)
        
        self.logger.info("Detecting outliers...")
        outliers = self.detect_outliers(answers_by_lang)
        
        self.logger.info("Computing n-grams...")
        bigram_stats = self.compute_ngram_analysis(answers_by_lang, n=2)
        trigram_stats = self.compute_ngram_analysis(answers_by_lang, n=3)
        
        self.logger.info("Generating flags...")
        flags = self.generate_flags(length_stats, vocab_stats, test_results)
        
        summary = {
            'total_responses': sum(len(v) for v in answers_by_lang.values()),
            'languages_analyzed': list(answers_by_lang.keys()),
            'significant_differences': sum(1 for t in test_results if t['Significant']),
            'outliers_detected': len(outliers),
            'flags_generated': len(flags),
            'critical_flags': sum(1 for f in flags if f.get('Severity') == 'Critical'),
            'high_flags': sum(1 for f in flags if f.get('Severity') == 'High')
        }
        
        results = {
            'response_length_stats': length_stats,
            'vocabulary_richness': vocab_stats,
            'bigram_analysis': bigram_stats,
            'trigram_analysis': trigram_stats,
            'statistical_tests': test_results,
            'outliers': outliers[:100],
            'flags': flags,
            'summary': summary
        }
        
        self.logger.info("="*50)
        self.logger.info("STATISTICAL AUDIT COMPLETE")
        self.logger.info(f"  Languages: {summary['languages_analyzed']}")
        self.logger.info(f"  Total responses: {summary['total_responses']}")
        self.logger.info(f"  Significant differences: {summary['significant_differences']}")
        self.logger.info(f"  Flags generated: {summary['flags_generated']}")
        self.logger.info("="*50)
        
        return results