"""
MaHealthBiasAudit - Stratum II: Linguistic Bias Audit
Linguistic analysis focusing on tokenisation and structural biases
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import PRIMARY_LANGUAGES, MAX_TOKEN_FERTILITY, SENTENCE_PIECING_THRESHOLD, WORD_PIECING_THRESHOLD
from utils import setup_logger, basic_tokenize


class LinguisticBiasAuditor:
    """Linguistic analysis for bias detection"""
    
    def __init__(self):
        self.logger = setup_logger('linguistic')
    
    def analyse_tokeniser_performance(self, 
                                      tokeniser_perfs: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Analyse tokeniser performance across languages"""
        rows = []
        
        for lang, perfs in tokeniser_perfs.items():
            for tokeniser, fertility in perfs.items():
                rows.append({
                    'Language': lang,
                    'Tokeniser': tokeniser,
                    'Fertility_Penalty': fertility,
                    'Is_Problematic': fertility > MAX_TOKEN_FERTILITY,
                    'Severity': 'High' if fertility > 1.8 else 'Moderate' if fertility > MAX_TOKEN_FERTILITY else 'Low'
                })
        
        return pd.DataFrame(rows)
    
    def analyse_subword_piecing(self, 
                               sample_words: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyse subword piecing patterns"""
        results = []
        
        for lang, words in sample_words.items():
            if lang == 'English':
                continue
            
            total_chars = 0
            total_subwords = 0
            
            for word in words[:100]:  # Sample
                total_chars += len(word)
                
                # Simulate subword splitting
                subwords = self._simulate_subword_split(word)
                total_subwords += len(subwords)
            
            avg_pieces_per_char = total_subwords / max(total_chars, 1)
            piecing_ratio = avg_pieces_per_char * 10  # Scale for readability
            
            results.append({
                'Language': lang,
                'Sample_Words_Analyzed': min(100, len(words)),
                'Avg_Subword_Pieces_Per_Word': total_subwords / max(min(100, len(words)), 1),
                'Piecing_Ratio': piecing_ratio,
                'Is_Highly_Pieced': piecing_ratio > SENTENCE_PIECING_THRESHOLD
            })
        
        return pd.DataFrame(results)
    
    def _simulate_subword_split(self, word: str) -> List[str]:
        """Simulate subword splitting for a word"""
        # Simplified simulation based on word length
        if len(word) <= 4:
            return [word]
        
        # Split at natural boundaries (common prefixes/suffixes)
        prefixes = ['mu', 'ku', 'ni', 'tu', 'ba', 'wa', 'ki', 'vi', 'a', 'e', 'i']
        suffixes = ['a', 'e', 'i', 'o', 'u', 'ka', 'ta', 'na', 'za', 'ya']
        
        result = []
        remaining = word
        
        # Check for prefix
        for pref in prefixes:
            if remaining.startswith(pref) and len(pref) < len(remaining):
                result.append(pref)
                remaining = remaining[len(pref):]
                break
        
        # Split remaining by length
        while len(remaining) > 4:
            # Look for common suffix
            found = False
            for suff in suffixes:
                if remaining.endswith(suff) and len(remaining) - len(suff) >= 3:
                    result.append(remaining[:-len(suff)])
                    result.append(suff)
                    remaining = ""
                    found = True
                    break
            
            if not found:
                split_point = min(3, len(remaining) - 1)
                result.append(remaining[:split_point])
                remaining = remaining[split_point:]
        
        if remaining:
            result.append(remaining)
        
        return result if len(result) > 1 else [word]
    
    def analyse_structural_complexity(self, 
                                     normalized_texts: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyse structural complexity of responses"""
        results = []
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in normalized_texts:
                continue
            
            texts = normalized_texts[lang]
            complexities = []
            avg_sentence_lengths = []
            avg_word_lengths = []
            
            for text in texts:
                sentences = text.split('.')
                sentences = [s.strip() for s in sentences if s.strip()]
                
                if sentences:
                    avg_sentence_lengths.append(np.mean([len(s.split()) for s in sentences]))
                
                words = basic_tokenize(text)
                if words:
                    avg_word_lengths.append(np.mean([len(w) for w in words]))
                
                # Complexity proxy: ratio of unique structures
                unique_bigrams = set(zip(words[:-1], words[1:]))
                total_bigrams = max(len(words) - 1, 1)
                complexities.append(len(unique_bigrams) / total_bigrams)
            
            results.append({
                'Language': lang,
                'Structural_Complexity_Mean': np.mean(complexities) if complexities else 0,
                'Structural_Complexity_Std': np.std(complexities) if complexities else 0,
                'Avg_Sentence_Length': np.mean(avg_sentence_lengths) if avg_sentence_lengths else 0,
                'Avg_Word_Length': np.mean(avg_word_lengths) if avg_word_lengths else 0
            })
        
        return pd.DataFrame(results)
    
    def compute_trust_aware_metrics(self, 
                                   questions_by_lang: Dict[str, List[str]],
                                   answers_by_lang: Dict[str, List[str]]) -> List[Dict]:
        """Compute trust-aware metrics comparing response quality"""
        results = []
        
        # Question-answer alignment check
        for lang in PRIMARY_LANGUAGES:
            if lang not in answers_by_lang or lang not in questions_by_lang:
                continue
            
            questions = questions_by_lang[lang]
            answers = answers_by_lang[lang]
            
            # Simplified alignment: check if answers contain keywords from questions
            alignments = []
            for q, a in zip(questions, answers):
                q_words = set(basic_tokenize(q))
                a_words = set(basic_tokenize(a))
                
                if q_words:
                    overlap = len(q_words.intersection(a_words)) / len(q_words)
                    alignments.append(overlap)
            
            # Also compute answer diversity
            answer_starts = [a[:30] if len(a) > 30 else a for a in answers]
            unique_starts = len(set(answer_starts))
            diversity_ratio = unique_starts / max(len(answers), 1)
            
            results.append({
                'Language': lang,
                'Question_Answer_Alignment': np.mean(alignments) if alignments else 0,
                'Answer_Diversity_Ratio': diversity_ratio,
                'Total_Answers': len(answers)
            })
        
        # Compare against English baseline
        english_alignment = next((r['Question_Answer_Alignment'] for r in results if r['Language'] == 'English'), 0)
        
        trust_aware_results = []
        for r in results:
            if r['Language'] != 'English' and english_alignment > 0:
                alignment_ratio = r['Question_Answer_Alignment'] / english_alignment
                trust_aware_results.append({
                    'Comparison': f"{r['Language']} vs English",
                    'Alignment_Ratio': alignment_ratio,
                    'Trust_Level': 'High' if alignment_ratio > 0.8 else 'Medium' if alignment_ratio > 0.6 else 'Low',
                    'Interpretation': f"{r['Language']} responses show {alignment_ratio:.1%} alignment compared to English"
                })
        
        return trust_aware_results
    
    def analyse_content_bias(self, 
                            normalized_texts: Dict[str, List[str]],
                            keywords: Dict[str, Dict[str, List[str]]]) -> pd.DataFrame:
        """Analyse content bias across languages using domain keywords"""
        results = []
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in normalized_texts:
                continue
            
            texts = normalized_texts[lang]
            lang_keywords = keywords.get(lang, {})
            
            content_scores = {}
            for category, kw_list in lang_keywords.items():
                if not kw_list:
                    continue
                
                scores = []
                for text in texts:
                    text_lower = text.lower()
                    count = sum(1 for kw in kw_list if kw in text_lower)
                    scores.append(count / max(len(kw_list), 1))
                
                content_scores[category] = np.mean(scores) if scores else 0
            
            # Calculate distribution
            if content_scores:
                total = sum(content_scores.values())
                if total > 0:
                    content_scores = {k: v/total for k, v in content_scores.items()}
            
            results.append({
                'Language': lang,
                **content_scores
            })
        
        return pd.DataFrame(results).fillna(0)
    
    def detect_linguistic_flags(self, 
                                tokeniser_df: pd.DataFrame,
                                complexity_df: pd.DataFrame,
                                trust_metrics: List[Dict]) -> List[Dict]:
        """Generate flags for linguistic biases"""
        flags = []
        
        # Tokeniser flags
        for _, row in tokeniser_df.iterrows():
            if row.get('Is_Problematic', False):
                flags.append({
                    'Type': 'Tokenisation_Bias',
                    'Language': row['Language'],
                    'Tokeniser': row['Tokeniser'],
                    'Severity': row.get('Severity', 'High'),
                    'Description': f"High fertility penalty ({row['Fertility_Penalty']:.2f}) for {row['Tokeniser']}",
                    'Recommendation': 'Consider using a different tokeniser for this language'
                })
        
        # Structural flags
        english_complexity = complexity_df[complexity_df['Language'] == 'English']['Structural_Complexity_Mean'].values
        if len(english_complexity) > 0:
            english_complexity = english_complexity[0]
            for _, row in complexity_df.iterrows():
                if row['Language'] != 'English':
                    ratio = row['Structural_Complexity_Mean'] / max(english_complexity, 0.001)
                    if ratio < 0.6:
                        flags.append({
                            'Type': 'Structural_Bias',
                            'Language': row['Language'],
                            'Severity': 'High',
                            'Description': f"Lower structural complexity ({ratio:.1%} of English)",
                            'Recommendation': 'Review if simpler responses indicate content loss'
                        })
        
        # Trust-aware flags
        for metric in trust_metrics:
            if metric['Trust_Level'] == 'Low':
                flags.append({
                    'Type': 'Trust_Bias',
                    'Comparison': metric['Comparison'],
                    'Severity': 'High',
                    'Description': metric['Interpretation'],
                    'Recommendation': 'Review translation quality and cultural adaptation'
                })
        
        return flags
    
    def run_full_audit(self,
                      questions_by_lang: Dict[str, List[str]],
                      answers_by_lang: Dict[str, List[str]],
                      tokeniser_perfs: Dict[str, Dict[str, float]],
                      sample_words: Dict[str, List[str]]) -> Dict:
        """Run complete linguistic bias audit"""
        self.logger.info("Starting linguistic bias audit")
        
        # Analyse tokeniser performance
        tokeniser_df = self.analyse_tokeniser_performance(tokeniser_perfs)
        
        # Analyse subword piecing
        piecing_df = self.analyse_subword_piecing(sample_words)
        
        # Analyse structural complexity
        complexity_df = self.analyse_structural_complexity(answers_by_lang)
        
        # Compute trust-aware metrics
        trust_metrics = self.compute_trust_aware_metrics(questions_by_lang, answers_by_lang)
        
        # Analyse content bias
        from config import DOMAIN_KEYWORDS
        content_df = self.analyse_content_bias(answers_by_lang, DOMAIN_KEYWORDS)
        
        # Generate flags
        flags = self.detect_linguistic_flags(tokeniser_df, complexity_df, trust_metrics)
        
        # Summary
        summary = {
            'tokenisers_analyzed': len(tokeniser_df['Tokeniser'].unique()) if not tokeniser_df.empty else 0,
            'problematic_tokenisers': len(tokeniser_df[tokeniser_df.get('Is_Problematic', False)]),
            'high_piecing_languages': len(piecing_df[piecing_df.get('Is_Highly_Pieced', False)]),
            'trust_issues': len([m for m in trust_metrics if m['Trust_Level'] == 'Low']),
            'flags_generated': len(flags)
        }
        
        results = {
            'tokeniser_performance': tokeniser_df,
            'subword_piecing_analysis': piecing_df,
            'structural_complexity': complexity_df,
            'trust_aware_results': trust_metrics,
            'content_analysis': content_df,
            'flags': flags,
            'summary': summary
        }
        
        self.logger.info(f"Linguistic audit complete: {summary}")
        return results