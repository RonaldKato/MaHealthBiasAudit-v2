"""
Stratum I: Statistical and Distributional Bias Audit
Based on Section 4 of the research proposal
Computes corpus statistics, Information Specific Divergence (ISD),
frequency distributions, and representation disparities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass
from scipy.stats import ks_2samp, chi2_contingency, entropy
import warnings
warnings.filterwarnings('ignore')

from config import MATERNAL_TOPICS, THRESHOLDS, PRIMARY_LANGUAGES
from utils import (
    compute_jensen_shannon_divergence, compute_type_token_ratio,
    compute_hapax_legomena, set_seed, RANDOM_SEED
)


@dataclass
class CorpusStatistics:
    """Comprehensive corpus statistics container"""
    language: str
    total_questions: int
    total_answers: int
    vocab_size: int
    type_token_ratio: float
    avg_question_length: float
    std_question_length: float
    avg_answer_length: float
    std_answer_length: float
    topic_distribution: Dict[str, float]
    lexical_diversity: float
    hapax_proportion: float
    morphological_complexity: float


@dataclass
class InformationSpecificDivergence:
    """Information Specific Divergence (ISD) results"""
    language_pair: Tuple[str, str]
    isd_value: float
    interpretation: str
    needs_intervention: bool


class StatisticalBiasAuditor:
    """
    Stratum I: Statistical bias audit
    Computes corpus statistics, JSD/ISD, frequency distributions,
    and representation disparities across languages
    """
    
    def __init__(self):
        """Initialize statistical bias auditor"""
        set_seed(RANDOM_SEED)
        self.corpus_stats: Dict[str, CorpusStatistics] = {}
        self.pairwise_jsd: Dict[str, float] = {}
        self.pairwise_isd: List[InformationSpecificDivergence] = []
        self.pairwise_ks: Dict[str, Dict] = {}
        self.bias_report_df = None
        self.frequency_distributions: Dict[str, Counter] = {}
    
    def compute_corpus_statistics(self, 
                                   questions: Dict[str, List[str]],
                                   answers: Dict[str, List[str]]) -> Dict[str, CorpusStatistics]:
        """
        Compute per-language corpus statistics:
        - Token count, type count, Type-Token Ratio (TTR)
        - Question and answer length distributions
        - Topic coverage (via keyword matching)
        - Hapax legomena proportions
        """
        print("\n" + "="*60)
        print("📊 Computing Corpus Statistics")
        print("="*60)
        
        for lang in questions.keys():
            q_texts = questions.get(lang, [])
            a_texts = answers.get(lang, [])
            
            if not q_texts:
                print(f"   ⚠️ No questions found for {lang}")
                continue
            
            # Combine texts for vocabulary analysis
            all_text = ' '.join(q_texts) + ' ' + ' '.join(a_texts)
            tokens = all_text.lower().split()
            
            vocab = set(tokens)
            ttr = len(vocab) / max(len(tokens), 1)
            
            # Hapax legomena (words appearing exactly once)
            hapax_count, hapax_prop = compute_hapax_legomena(tokens)
            
            # Question lengths
            q_lengths = [len(q.split()) for q in q_texts if q]
            avg_q_len = np.mean(q_lengths) if q_lengths else 0
            std_q_len = np.std(q_lengths) if q_lengths else 0
            
            # Answer lengths
            a_lengths = [len(a.split()) for a in a_texts if a]
            avg_a_len = np.mean(a_lengths) if a_lengths else 0
            std_a_len = np.std(a_lengths) if a_lengths else 0
            
            # Topic distribution
            topic_dist = self._compute_topic_distribution(q_texts + a_texts)
            
            # Morphological complexity from config
            from config import LANGUAGES
            morph_complexity = LANGUAGES.get(lang, {}).get('morphological_complexity', 1.0)
            
            stats = CorpusStatistics(
                language=lang,
                total_questions=len(q_texts),
                total_answers=len(a_texts),
                vocab_size=len(vocab),
                type_token_ratio=ttr,
                avg_question_length=avg_q_len,
                std_question_length=std_q_len,
                avg_answer_length=avg_a_len,
                std_answer_length=std_a_len,
                topic_distribution=topic_dist,
                lexical_diversity=ttr,
                hapax_proportion=hapax_prop,
                morphological_complexity=morph_complexity
            )
            self.corpus_stats[lang] = stats
            
            print(f"   {lang}: TTR={ttr:.3f}, Vocab={len(vocab)}, "
                  f"Hapax={hapax_prop:.3f}, Q_len={avg_q_len:.1f}")
        
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
    
    def compute_frequency_distributions(self, 
                                        texts: Dict[str, List[str]]) -> Dict[str, Counter]:
        """
        Compute token frequency distributions for each language
        Used for Information Specific Divergence (ISD) calculation
        """
        print("\n" + "="*60)
        print("📈 Computing Frequency Distributions")
        print("="*60)
        
        for lang, text_list in texts.items():
            all_tokens = []
            for text in text_list:
                all_tokens.extend(text.lower().split())
            
            self.frequency_distributions[lang] = Counter(all_tokens)
            
            print(f"   {lang}: {len(self.frequency_distributions[lang])} unique tokens, "
                  f"{sum(self.frequency_distributions[lang].values())} total tokens")
        
        return self.frequency_distributions
    
    def compute_information_specific_divergence(self) -> List[InformationSpecificDivergence]:
        """
        Compute Information Specific Divergence (ISD) between language pairs
        ISD measures the divergence in information content across languages
        Based on the research proposal: ISD(L_i, L_j) = KL(P_i || P_j) + KL(P_j || P_i)
        """
        print("\n" + "="*60)
        print("📐 Computing Information Specific Divergence (ISD)")
        print("="*60)
        
        languages = list(self.frequency_distributions.keys())
        self.pairwise_isd = []
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i >= j:
                    continue
                
                dist1 = self.frequency_distributions[lang1]
                dist2 = self.frequency_distributions[lang2]
                
                # Convert to probability distributions
                total1 = sum(dist1.values())
                total2 = sum(dist2.values())
                
                # Get union of tokens
                all_tokens = set(dist1.keys()) | set(dist2.keys())
                
                # Create aligned probability vectors with smoothing
                epsilon = 1e-12
                p1 = np.array([(dist1.get(tok, 0) + epsilon) / (total1 + epsilon * len(all_tokens)) 
                              for tok in all_tokens])
                p2 = np.array([(dist2.get(tok, 0) + epsilon) / (total2 + epsilon * len(all_tokens)) 
                              for tok in all_tokens])
                
                # Normalize
                p1 = p1 / np.sum(p1)
                p2 = p2 / np.sum(p2)
                
                # Compute KL divergences
                kl_12 = entropy(p1, p2)  # Using scipy's entropy which computes KL
                kl_21 = entropy(p2, p1)
                
                # ISD is the sum of both KL divergences
                isd_value = kl_12 + kl_21
                
                # Interpret the ISD value
                if isd_value > THRESHOLDS['jsd_high']:
                    interpretation = "High divergence - significant information disparity"
                    needs_intervention = True
                elif isd_value > THRESHOLDS['jsd_high'] / 2:
                    interpretation = "Moderate divergence - some information disparity"
                    needs_intervention = True
                else:
                    interpretation = "Low divergence - good information alignment"
                    needs_intervention = False
                
                isd_result = InformationSpecificDivergence(
                    language_pair=(lang1, lang2),
                    isd_value=isd_value,
                    interpretation=interpretation,
                    needs_intervention=needs_intervention
                )
                self.pairwise_isd.append(isd_result)
                
                print(f"   {lang1} ↔ {lang2}: ISD={isd_value:.4f} "
                      f"({'⚠️' if needs_intervention else '✓'})")
        
        return self.pairwise_isd
    
    def compute_jensen_shannon_divergence(self, 
                                           questions: Dict[str, List[str]]) -> Dict:
        """
        Compute Jensen-Shannon Divergence between language sub-corpora
        JSD > 0.5 indicates high representational divergence
        """
        print("\n" + "="*60)
        print("📊 Computing Jensen-Shannon Divergence")
        print("="*60)
        
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
                    jsd = compute_jensen_shannon_divergence(unigrams[lang1], unigrams[lang2])
                    self.pairwise_jsd[f"{lang1}_vs_{lang2}"] = jsd
                    valid_jsds.append(jsd)
                    
                    status = "HIGH" if jsd > THRESHOLDS['jsd_high'] else "MODERATE" if jsd > 0.3 else "LOW"
                    print(f"   {lang1} ↔ {lang2}: JSD={jsd:.4f} [{status}]")
                    
                except Exception as e:
                    print(f"   ⚠️ JSD calculation failed for {lang1}-{lang2}: {e}")
                    self.pairwise_jsd[f"{lang1}_vs_{lang2}"] = 0.0
        
        # Compute average JSD from valid values
        if valid_jsds:
            self.pairwise_jsd['average'] = np.mean(valid_jsds)
        else:
            self.pairwise_jsd['average'] = 0.0
        
        print(f"\n   Average JSD: {self.pairwise_jsd['average']:.4f}")
        
        return self.pairwise_jsd
    
    def compute_length_distribution_equivalence(self, 
                                                  questions: Dict[str, List[str]]) -> Dict:
        """
        Test for equivalence of question length distributions using K-S test
        """
        print("\n" + "="*60)
        print("📏 Testing Length Distribution Equivalence")
        print("="*60)
        
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
                        'different_distribution': p_value < 0.05,
                        'interpretation': 'Different' if p_value < 0.05 else 'Similar'
                    }
                    
                    status = "⚠️ DIFFERENT" if p_value < 0.05 else "✓ SIMILAR"
                    print(f"   {lang1} vs {lang2}: KS={ks_stat:.3f}, p={p_value:.4f} [{status}]")
        
        return self.pairwise_ks
    
    def compute_topic_coverage_homogeneity(self) -> Dict:
        """
        Compute topic coverage homogeneity using chi-squared test
        Measures representational disparity across languages
        """
        print("\n" + "="*60)
        print("🎯 Computing Topic Coverage Homogeneity")
        print("="*60)
        
        if len(self.corpus_stats) < 2:
            return {
                'chi2_statistic': 0,
                'p_value': 1.0,
                'degrees_of_freedom': 0,
                'cramers_v': 0,
                'significant_difference': False,
                'interpretation': 'Insufficient data for comparison'
            }
        
        # Get topics
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
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Cramer's V for effect size
            n = np.sum(contingency)
            min_dim = min(contingency.shape) - 1
            if min_dim > 0 and n > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
            else:
                cramers_v = 0
            
            interpretation = (
                "Significant topic coverage disparity" if cramers_v > 0.3 else
                "Moderate topic coverage difference" if cramers_v > 0.1 else
                "Similar topic coverage across languages"
            )
            
            print(f"\n   Chi-squared: {chi2:.2f}, p={p_value:.4f}")
            print(f"   Cramer's V: {cramers_v:.3f}")
            print(f"   Interpretation: {interpretation}")
            
            return {
                'chi2_statistic': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'cramers_v': cramers_v,
                'significant_difference': p_value < 0.05 if not np.isnan(p_value) else False,
                'interpretation': interpretation
            }
            
        except Exception as e:
            print(f"   ⚠️ Chi-squared test failed: {e}")
            return {
                'chi2_statistic': 0,
                'p_value': 1.0,
                'degrees_of_freedom': 0,
                'cramers_v': 0,
                'significant_difference': False,
                'interpretation': f'Test failed: {str(e)[:50]}'
            }
    
    def compute_representation_disparities(self) -> pd.DataFrame:
        """
        Compute representation disparities across languages
        Measures how equally different languages are represented
        """
        print("\n" + "="*60)
        print("⚖️ Computing Representation Disparities")
        print("="*60)
        
        disparities = []
        
        if not self.corpus_stats:
            return pd.DataFrame()
        
        # Find baseline (English as reference)
        eng_stats = self.corpus_stats.get('English')
        if not eng_stats:
            eng_stats = list(self.corpus_stats.values())[0]
            baseline_lang = list(self.corpus_stats.keys())[0]
        else:
            baseline_lang = 'English'
        
        baseline_vocab = eng_stats.vocab_size
        baseline_ttr = eng_stats.type_token_ratio
        baseline_hapax = eng_stats.hapax_proportion
        
        for lang, stats in self.corpus_stats.items():
            if lang == baseline_lang:
                continue
            
            vocab_ratio = stats.vocab_size / max(baseline_vocab, 1)
            ttr_ratio = stats.type_token_ratio / max(baseline_ttr, 1)
            hapax_ratio = stats.hapax_proportion / max(baseline_hapax, 1)
            
            # Composite disparity score
            disparity_score = (1 - vocab_ratio) + (1 - ttr_ratio) + abs(1 - hapax_ratio)
            disparity_score = disparity_score / 3  # Normalize to 0-1
            
            disparities.append({
                'Language': lang,
                'Baseline': baseline_lang,
                'Vocab_Ratio': vocab_ratio,
                'TTR_Ratio': ttr_ratio,
                'Hapax_Ratio': hapax_ratio,
                'Disparity_Score': disparity_score,
                'Severity': 'High' if disparity_score > 0.5 else 'Moderate' if disparity_score > 0.25 else 'Low'
            })
            
            print(f"   {lang} vs {baseline_lang}: Disparity={disparity_score:.3f} [{disparities[-1]['Severity']}]")
        
        return pd.DataFrame(disparities)
    
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
                'Hapax_Proportion': round(stats.hapax_proportion, 4),
                'Avg_Question_Length': round(stats.avg_question_length, 2),
                'Avg_Answer_Length': round(stats.avg_answer_length, 2),
                'Lexical_Diversity': round(stats.lexical_diversity, 4),
                'Morphological_Complexity': stats.morphological_complexity
            })
        
        self.bias_report_df = pd.DataFrame(rows)
        return self.bias_report_df
    
    def get_flags(self) -> List[str]:
        """Generate flags based on statistical bias thresholds"""
        flags = []
        
        # Check JSD/ISD thresholds
        for isd_result in self.pairwise_isd:
            if isd_result.needs_intervention:
                flags.append(f"HIGH_ISD: {isd_result.language_pair[0]}-{isd_result.language_pair[1]} = {isd_result.isd_value:.3f}")
        
        # Check KS test flags
        for pair, result in self.pairwise_ks.items():
            if result.get('different_distribution', False):
                flags.append(f"DIFFERENT_LENGTH_DIST: {pair}")
        
        # Check topic homogeneity
        topic_homog = self.compute_topic_coverage_homogeneity()
        if topic_homog.get('significant_difference', False):
            cramers_v = topic_homog.get('cramers_v', 0)
            if cramers_v > 0.3:
                flags.append(f"TOPIC_COVERAGE_DISPARITY: Cramer's V = {cramers_v:.3f}")
        
        # Check representation disparities
        disparity_df = self.compute_representation_disparities()
        if not disparity_df.empty:
            high_disparity = disparity_df[disparity_df['Severity'] == 'High']
            for _, row in high_disparity.iterrows():
                flags.append(f"REPRESENTATION_DISPARITY: {row['Language']} (score={row['Disparity_Score']:.3f})")
        
        return flags
    
    def run_full_audit(self, questions: Dict[str, List[str]], 
                        answers: Dict[str, List[str]]) -> Dict:
        """
        Run complete statistical bias audit
        
        Args:
            questions: Dictionary of questions per language
            answers: Dictionary of answers per language
        
        Returns:
            Dictionary with all audit results
        """
        print("\n" + "="*70)
        print("📊 STRATUM I: Statistical Bias Audit")
        print("="*70)
        
        # Compute corpus statistics
        corpus_stats = self.compute_corpus_statistics(questions, answers)
        
        # Compute frequency distributions
        freq_dists = self.compute_frequency_distributions(answers)
        
        # Compute Information Specific Divergence (ISD)
        isd_results = self.compute_information_specific_divergence()
        
        # Compute Jensen-Shannon Divergence
        jsd_results = self.compute_jensen_shannon_divergence(questions)
        
        # Test length distribution equivalence
        ks_results = self.compute_length_distribution_equivalence(questions)
        
        # Compute topic coverage homogeneity
        topic_homog = self.compute_topic_coverage_homogeneity()
        
        # Compute representation disparities
        rep_disparities = self.compute_representation_disparities()
        
        # Generate report
        report_df = self.generate_bias_report()
        flags = self.get_flags()
        
        return {
            'corpus_statistics': corpus_stats,
            'frequency_distributions': freq_dists,
            'information_specific_divergence': isd_results,
            'jensen_shannon_divergence': jsd_results,
            'ks_test_results': ks_results,
            'topic_coverage_homogeneity': topic_homog,
            'representation_disparities': rep_disparities,
            'bias_report': report_df,
            'flags': flags
        }


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
    
    results = auditor.run_full_audit(sample_questions, sample_answers)
    
    print("\n✅ Statistical bias audit complete!")
    print(f"   Flags generated: {len(results['flags'])}")