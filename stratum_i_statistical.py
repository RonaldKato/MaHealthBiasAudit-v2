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
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in normalized_texts:
                continue
            
            texts = normalized_texts[lang]
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
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in normalized_texts:
                continue
            
            texts = normalized_texts[lang]
            richness_scores = []
            diversity_scores = []
            
            for text in texts:
                tokens = basic_tokenize(text)
                if tokens:
                    richness_scores.append(compute_vocabulary_richness(tokens))
                    diversity_scores.append(compute_lexical_diversity(tokens))
            
            stats.append({
                'Language': lang,
                'Vocabulary_Richness_Mean': np.mean(richness_scores),
                'Vocabulary_Richness_Std': np.std(richness_scores),
                'Lexical_Diversity_Mean': np.mean(diversity_scores),
                'Lexical_Diversity_Std': np.std(diversity_scores)
            })
        
        return pd.DataFrame(stats)
    
    def compute_ngram_analysis(self, 
                               normalized_texts: Dict[str, List[str]], 
                               n: int = 2) -> pd.DataFrame:
        """Compute n-gram frequency analysis"""
        results = []
        
        vectorizer = CountVectorizer(ngram_range=(n, n), lowercase=True, 
                                     tokenizer=basic_tokenize, token_pattern=None)
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in normalized_texts:
                continue
            
            texts = normalized_texts[lang]
            if not texts:
                continue
            
            try:
                ngram_matrix = vectorizer.fit_transform(texts)
                ngram_counts = np.array(ngram_matrix.sum(axis=0)).flatten()
                ngram_names = vectorizer.get_feature_names_out()
                
                # Get top n-grams
                top_indices = np.argsort(ngram_counts)[-20:][::-1]
                top_ngrams = [(ngram_names[i], ngram_counts[i]) for i in top_indices]
                
                results.append({
                    'Language': lang,
                    f'Top_{n}-grams': top_ngrams[:10],
                    'Unique_ngrams': len(ngram_names),
                    'Total_ngram_occurrences': sum(ngram_counts)
                })
            except Exception as e:
                self.logger.warning(f"Error computing {n}-grams for {lang}: {e}")
                results.append({'Language': lang, f'Top_{n}-grams': [], 'Unique_ngrams': 0, 'Total_ngram_occurrences': 0})
        
        return pd.DataFrame(results)
    
    def perform_statistical_tests(self, 
                                  normalized_texts: Dict[str, List[str]]) -> List[Dict]:
        """Perform statistical tests comparing languages"""
        results = []
        
        # Prepare data
        lang_data = {}
        for lang in PRIMARY_LANGUAGES:
            if lang in normalized_texts:
                lengths = [len(t.split()) for t in normalized_texts[lang]]
                lang_data[lang] = lengths
        
        # Pairwise comparisons
        for i, lang1 in enumerate(PRIMARY_LANGUAGES):
            if lang1 not in lang_data:
                continue
            for lang2 in PRIMARY_LANGUAGES[i+1:]:
                if lang2 not in lang_data:
                    continue
                
                data1 = lang_data[lang1]
                data2 = lang_data[lang2]
                
                # Mann-Whitney U test (non-parametric)
                try:
                    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                except:
                    u_stat, p_value = 0, 1.0
                
                # Cohen's d effect size
                d = cohens_d(np.array(data1), np.array(data2))
                
                # Interpretation
                if p_value < SIGNIFICANCE_ALPHA:
                    if d > 0.5:
                        interpretation = f"Large effect: {lang1} responses are significantly {'longer' if np.mean(data1) > np.mean(data2) else 'shorter'}"
                    elif d > 0.2:
                        interpretation = f"Medium effect: {lang1} responses are significantly {'longer' if np.mean(data1) > np.mean(data2) else 'shorter'}"
                    else:
                        interpretation = f"Small effect: {lang1} responses are significantly {'longer' if np.mean(data1) > np.mean(data2) else 'shorter'}"
                else:
                    interpretation = "No significant difference"
                
                results.append({
                    'Comparison': f"{lang1} vs {lang2}",
                    'Mean_Diff': np.mean(data1) - np.mean(data2),
                    'Mann_Whitney_U': u_stat,
                    'P_Value': p_value,
                    'Significant': p_value < SIGNIFICANCE_ALPHA,
                    'Effect_Size_Cohens_d': d,
                    'Interpretation': interpretation
                })
        
        return results
    
    def detect_outliers(self, normalized_texts: Dict[str, List[str]]) -> List[Dict]:
        """Detect outlier responses that may indicate bias"""
        outliers = []
        
        for lang, texts in normalized_texts.items():
            lengths = [len(t.split()) for t in texts]
            if not lengths:
                continue
            
            # IQR method
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
                        'Text_Preview': texts[idx][:200] if len(texts[idx]) > 200 else texts[idx],
                        'Outlier_Type': 'Short' if length < lower_bound else 'Long'
                    })
        
        return outliers
    
    def compute_category_analysis(self, 
                                  normalized_texts: Dict[str, List[str]],
                                  category_splits: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
        """Analyse bias across different categories"""
        results = []
        
        for category, lang_texts in category_splits.items():
            for lang in PRIMARY_LANGUAGES:
                if lang not in lang_texts:
                    continue
                
                texts = lang_texts[lang]
                lengths = [len(t.split()) for t in texts]
                
                results.append({
                    'Category': category,
                    'Language': lang,
                    'Avg_Length': np.mean(lengths) if lengths else 0,
                    'Count': len(texts)
                })
        
        return pd.DataFrame(results)
    
    def generate_flags(self, 
                       length_stats: pd.DataFrame,
                       vocab_stats: pd.DataFrame,
                       test_results: List[Dict]) -> List[Dict]:
        """Generate flags for potential biases"""
        flags = []
        
        # Length bias detection
        english_mean = length_stats[length_stats['Language'] == 'English']['Mean'].values
        if len(english_mean) > 0:
            english_mean = english_mean[0]
            for _, row in length_stats.iterrows():
                if row['Language'] != 'English':
                    ratio = row['Mean'] / english_mean
                    if ratio < 0.6:
                        flags.append({
                            'Type': 'Length_Bias',
                            'Language': row['Language'],
                            'Severity': 'High',
                            'Description': f"Responses are {ratio:.1%} of English length ({row['Mean']:.0f} vs {english_mean:.0f} words)",
                            'Recommendation': 'Review translation quality and cultural adaptation'
                    })
                    elif ratio < 0.8:
                        flags.append({
                            'Type': 'Length_Bias',
                            'Language': row['Language'],
                            'Severity': 'Moderate',
                            'Description': f"Responses are {ratio:.1%} of English length ({row['Mean']:.0f} vs {english_mean:.0f} words)",
                            'Recommendation': 'Consider if content loss is occurring'
                    })
        
        # Vocabulary richness bias
        for _, row in vocab_stats.iterrows():
            if row['Language'] != 'English' and row['Vocabulary_Richness_Mean'] < 0.3:
                flags.append({
                    'Type': 'Vocabulary_Bias',
                    'Language': row['Language'],
                    'Severity': 'High',
                    'Description': f"Low vocabulary richness ({row['Vocabulary_Richness_Mean']:.3f})",
                    'Recommendation': 'Check if data collection captures natural language variation'
                })
        
        # Statistical significance flags
        for test in test_results:
            if test['Significant'] and abs(test['Effect_Size_Cohens_d']) > 0.5:
                flags.append({
                    'Type': 'Statistical_Bias',
                    'Comparison': test['Comparison'],
                    'Severity': 'High',
                    'Description': test['Interpretation'],
                    'Recommendation': 'Investigate potential systematic differences'
                })
        
        return flags
    
    def run_full_audit(self, 
                       questions_by_lang: Dict[str, List[str]], 
                       answers_by_lang: Dict[str, List[str]]) -> Dict:
        """Run complete statistical bias audit"""
        self.logger.info("Starting statistical bias audit")
        
        # Compute statistics
        length_stats = self.compute_response_length_stats(answers_by_lang)
        vocab_stats = self.compute_vocabulary_richness_stats(answers_by_lang)
        
        # Perform statistical tests
        test_results = self.perform_statistical_tests(answers_by_lang)
        
        # Detect outliers
        outliers = self.detect_outliers(answers_by_lang)
        
        # N-gram analysis
        bigram_stats = self.compute_ngram_analysis(answers_by_lang, n=2)
        trigram_stats = self.compute_ngram_analysis(answers_by_lang, n=3)
        
        # Generate flags
        flags = self.generate_flags(length_stats, vocab_stats, test_results)
        
        # Prepare summary
        summary = {
            'total_responses': sum(len(v) for v in answers_by_lang.values()),
            'languages_analyzed': [l for l in PRIMARY_LANGUAGES if l in answers_by_lang],
            'significant_differences': sum(1 for t in test_results if t['Significant']),
            'outliers_detected': len(outliers),
            'flags_generated': len(flags)
        }
        
        results = {
            'response_length_stats': length_stats,
            'vocabulary_richness': vocab_stats,
            'bigram_analysis': bigram_stats,
            'trigram_analysis': trigram_stats,
            'statistical_tests': test_results,
            'outliers': outliers[:50],  # Limit outliers
            'flags': flags,
            'summary': summary
        }
        
        self.logger.info(f"Statistical audit complete: {summary}")
        return results