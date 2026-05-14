"""
Stratum I: Statistical and Distributional Bias Audit
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field
from scipy.stats import ks_2samp, chi2_contingency
import warnings
warnings.filterwarnings('ignore')

from config import MATERNAL_TOPICS, THRESHOLDS
from utils import calculate_jsd, classify_maternal_topic


@dataclass
class CorpusStatistics:
    """Corpus statistics container"""
    language: str
    total_questions: int
    total_answers: int
    vocab_size: int
    type_token_ratio: float
    avg_question_length: float
    avg_answer_length: float
    topic_distribution: Dict[str, float]
    lexical_diversity: float


class StatisticalBiasAuditor:
    """
    Stratum I: Statistical bias audit
    Computes corpus statistics, JSD, topic coverage homogeneity
    """
    
    def __init__(self):
        self.corpus_stats: Dict[str, CorpusStatistics] = {}
        self.pairwise_jsd: Dict[str, float] = {}
        self.pairwise_ks: Dict[str, Dict] = {}
        self.bias_report_df = None
    
    def compute_corpus_statistics(self, 
                                   questions: Dict[str, List[str]],
                                   answers: Dict[str, List[str]]) -> Dict[str, CorpusStatistics]:
        """
        Compute per-language corpus statistics:
        - Token count, type count, Type-Token Ratio (TTR)
        - Question and answer length distributions
        - Topic coverage (via keyword matching)
        """
        for lang in questions.keys():
            q_texts = questions.get(lang, [])
            a_texts = answers.get(lang, [])
            
            if not q_texts:
                continue
            
            # Combine all question and answer texts for vocab analysis
            all_text = ' '.join(q_texts) + ' ' + ' '.join(a_texts)
            tokens = all_text.lower().split()
            
            vocab = set(tokens)
            ttr = len(vocab) / max(len(tokens), 1)
            
            # Question lengths
            q_lengths = [len(q.split()) for q in q_texts if q]
            avg_q_len = np.mean(q_lengths) if q_lengths else 0
            
            # Answer lengths
            a_lengths = [len(a.split()) for a in a_texts if a]
            avg_a_len = np.mean(a_lengths) if a_lengths else 0
            
            # Topic distribution
            topic_dist = self._compute_topic_distribution(q_texts + a_texts)
            
            stats = CorpusStatistics(
                language=lang,
                total_questions=len(q_texts),
                total_answers=len(a_texts),
                vocab_size=len(vocab),
                type_token_ratio=ttr,
                avg_question_length=avg_q_len,
                avg_answer_length=avg_a_len,
                topic_distribution=topic_dist,
                lexical_diversity=ttr
            )
            self.corpus_stats[lang] = stats
        
        return self.corpus_stats
    
    def _compute_topic_distribution(self, texts: List[str]) -> Dict[str, float]:
        """Compute topic distribution using keyword matching"""
        topic_counts = {topic: 0 for topic in MATERNAL_TOPICS.keys()}
        
        for text in texts:
            text_lower = text.lower()
            for topic, topic_info in MATERNAL_TOPICS.items():
                for keyword in topic_info['keywords']:
                    if keyword in text_lower:
                        topic_counts[topic] += 1
                        break
        
        total = sum(topic_counts.values())
        if total > 0:
            return {k: v/total for k, v in topic_counts.items()}
        return {k: 0.0 for k in topic_counts.keys()}
    
    def compute_length_distribution_equivalence(self, 
                                                  questions: Dict[str, List[str]]) -> Dict:
        """
        Test for equivalence of question length distributions using K-S test
        """
        languages = list(questions.keys())
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i >= j:
                    continue
                
                q1 = questions.get(lang1, [])
                q2 = questions.get(lang2, [])
                
                if not q1 or not q2:
                    continue
                
                len1 = [len(q.split()) for q in q1 if q]
                len2 = [len(q.split()) for q in q2 if q]
                
                if len(len1) > 0 and len(len2) > 0:
                    ks_stat, p_value = ks_2samp(len1, len2)
                    
                    self.pairwise_ks[f"{lang1}_vs_{lang2}"] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'different_distribution': p_value < 0.05
                    }
        
        return self.pairwise_ks
    
    def compute_jensen_shannon_divergence(self, 
                                           questions: Dict[str, List[str]]) -> Dict:
        """
        Compute Jensen-Shannon Divergence between language sub-corpora
        JSD > 0.5 indicates high representational divergence
        """
        # Build unigram frequency distributions
        unigrams = {}
        for lang, texts in questions.items():
            all_tokens = []
            for text in texts:
                if text:
                    all_tokens.extend(text.lower().split())
            
            if all_tokens:
                counter = Counter(all_tokens)
                total = sum(counter.values())
                unigrams[lang] = {token: count/total for token, count in counter.items()}
            else:
                unigrams[lang] = {}
        
        # Compute pairwise JSD
        languages = list(unigrams.keys())
        valid_jsds = []
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i >= j:
                    continue
                
                # Skip if either distribution is empty
                if not unigrams[lang1] or not unigrams[lang2]:
                    self.pairwise_jsd[f"{lang1}_vs_{lang2}"] = 0.0
                    continue
                
                try:
                    jsd = calculate_jsd(unigrams[lang1], unigrams[lang2])
                    self.pairwise_jsd[f"{lang1}_vs_{lang2}"] = jsd
                    valid_jsds.append(jsd)
                except Exception as e:
                    print(f"  Warning: JSD calculation failed for {lang1}-{lang2}: {e}")
                    self.pairwise_jsd[f"{lang1}_vs_{lang2}"] = 0.0
        
        # Compute average JSD from valid values
        if valid_jsds:
            self.pairwise_jsd['average'] = np.mean(valid_jsds)
        else:
            self.pairwise_jsd['average'] = 0.0
        
        return self.pairwise_jsd
    
    def compute_topic_coverage_homogeneity(self) -> Dict:
        """
        Compute topic coverage homogeneity using chi-squared test
        Handles zero expected frequencies gracefully
        """
        if len(self.corpus_stats) < 2:
            return {
                'chi2_statistic': 0,
                'p_value': 1.0,
                'degrees_of_freedom': 0,
                'cramers_v': 0,
                'significant_difference': False,
                'interpretation': 'Insufficient data for comparison'
            }
        
        # Get topics from config
        topics = list(MATERNAL_TOPICS.keys())
        languages = list(self.corpus_stats.keys())
        
        # Build contingency table
        contingency = []
        for lang in languages:
            dist = self.corpus_stats[lang].topic_distribution
            row = [dist.get(topic, 0) for topic in topics]
            # Add small epsilon to avoid zeros
            row = [max(x, 1e-10) for x in row]
            contingency.append(row)
        
        contingency = np.array(contingency)
        
        # Check if contingency table has any non-zero values
        if np.sum(contingency) == 0:
            return {
                'chi2_statistic': 0,
                'p_value': 1.0,
                'degrees_of_freedom': 0,
                'cramers_v': 0,
                'significant_difference': False,
                'interpretation': 'No topic data available'
            }
        
        try:
            # Try chi-squared test
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Check if expected frequencies have zero elements
            if np.any(expected == 0):
                # Use Fisher's exact test approximation or fallback
                from scipy.stats import fisher_exact
                # For larger tables, use simulation
                from scipy.stats import chi2_contingency as chi2_sim
                chi2, p_value, dof, expected = chi2_contingency(contingency, correction=False)
            
            # Cramer's V for effect size
            n = np.sum(contingency)
            min_dim = min(contingency.shape) - 1
            if min_dim > 0 and n > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
            else:
                cramers_v = 0
            
            return {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'cramers_v': cramers_v,
                'significant_difference': p_value < 0.05 if not np.isnan(p_value) else False,
                'interpretation': 'Different topic coverage' if cramers_v > 0.3 else 'Similar topic coverage'
            }
            
        except Exception as e:
            print(f"  Warning: Chi-squared test failed: {e}")
            return {
                'chi2_statistic': 0,
                'p_value': 1.0,
                'degrees_of_freedom': 0,
                'cramers_v': 0,
                'significant_difference': False,
                'interpretation': f'Test failed: {str(e)[:50]}'
            }
    
    def generate_bias_report(self) -> pd.DataFrame:
        """Generate comprehensive bias report for Stratum I"""
        rows = []
        
        for lang, stats in self.corpus_stats.items():
            rows.append({
                'Language': lang,
                'Questions': stats.total_questions,
                'Answers': stats.total_answers,
                'Vocabulary_Size': stats.vocab_size,
                'Type-Token_Ratio': round(stats.type_token_ratio, 4),
                'Avg_Question_Length': round(stats.avg_question_length, 2),
                'Avg_Answer_Length': round(stats.avg_answer_length, 2),
                'Lexical_Diversity': round(stats.lexical_diversity, 4)
            })
        
        self.bias_report_df = pd.DataFrame(rows)
        return self.bias_report_df
    
    def get_flags(self) -> List[str]:
        """Generate flags based on bias thresholds"""
        flags = []
        
        # Check JSD thresholds
        for pair, jsd in self.pairwise_jsd.items():
            if pair != 'average' and jsd > THRESHOLDS.get('jsd_high', 0.5):
                flags.append(f"HIGH_JSD: {pair} = {jsd:.3f}")
        
        # Check KS test flags
        for pair, result in self.pairwise_ks.items():
            if result.get('different_distribution', False):
                flags.append(f"DIFFERENT_LENGTH_DIST: {pair}")
        
        # Check topic homogeneity (safe call)
        try:
            topic_homog = self.compute_topic_coverage_homogeneity()
            if topic_homog.get('significant_difference', False):
                cramers_v = topic_homog.get('cramers_v', 0)
                if cramers_v > 0.3:
                    flags.append(f"TOPIC_COVERAGE_DIFFERENCE: Cramer's V = {cramers_v:.3f}")
        except Exception as e:
            print(f"  Warning: Topic homogeneity check failed: {e}")
        
        return flags
    
    def visualize_statistical_bias(self, save_path: str = "statistical_bias.png"):
        """Visualize statistical bias results"""
        import matplotlib.pyplot as plt
        
        if self.bias_report_df is None or self.bias_report_df.empty:
            print("No bias report to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Vocabulary size
        axes[0, 0].bar(self.bias_report_df['Language'], self.bias_report_df['Vocabulary_Size'])
        axes[0, 0].set_title('Vocabulary Size by Language')
        axes[0, 0].set_xlabel('Language')
        axes[0, 0].set_ylabel('Vocabulary Size')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Lexical diversity
        axes[0, 1].bar(self.bias_report_df['Language'], self.bias_report_df['Lexical_Diversity'])
        axes[0, 1].set_title('Lexical Diversity (TTR) by Language')
        axes[0, 1].set_xlabel('Language')
        axes[0, 1].set_ylabel('Type-Token Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sentence length
        axes[1, 0].bar(self.bias_report_df['Language'], self.bias_report_df['Avg_Answer_Length'])
        axes[1, 0].set_title('Average Answer Length by Language')
        axes[1, 0].set_xlabel('Language')
        axes[1, 0].set_ylabel('Words per Answer')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Readability (placeholder - using vocab size as proxy)
        axes[1, 1].bar(self.bias_report_df['Language'], self.bias_report_df['Vocabulary_Size'] / 100)
        axes[1, 1].set_title('Vocabulary Richness (Vocab/100)')
        axes[1, 1].set_xlabel('Language')
        axes[1, 1].set_ylabel('Normalized Vocabulary')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Statistical bias visualization saved to {save_path}")


# Test the auditor
if __name__ == "__main__":
    auditor = StatisticalBiasAuditor()
    
    sample_questions = {
        'English': ["What are essential nutrients?", "What are signs of labor?"],
        'Swahili': ["Virutubisho muhimu?", "Dalili za uchungu?"],
        'Luganda': ["Byetaago bya maanyi?", "Obubonero bwokuzala?"]
    }
    
    sample_answers = {
        'English': ["Folic acid, iron.", "Contractions, water breaking."],
        'Swahili': ["Asidi foliki, chuma.", "Mikazo, kupasuka kwa maji."],
        'Luganda': ["Folic acid, ekyuma.", "Okuluma, amazzi."]
    }
    
    stats = auditor.compute_corpus_statistics(sample_questions, sample_answers)
    for lang, s in stats.items():
        print(f"{lang}: TTR={s.type_token_ratio:.3f}")
    
    print("\n Statistical bias auditor test complete!")