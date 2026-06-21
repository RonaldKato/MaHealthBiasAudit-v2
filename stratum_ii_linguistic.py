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

from config import PRIMARY_LANGUAGES, MAX_TOKEN_FERTILITY, SENTENCE_PIECING_THRESHOLD, DOMAIN_KEYWORDS
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
                if fertility > 1.8:
                    severity = 'Critical'
                elif fertility > MAX_TOKEN_FERTILITY:
                    severity = 'High'
                elif fertility > 1.3:
                    severity = 'Moderate'
                else:
                    severity = 'Low'
                
                rows.append({
                    'Language': lang,
                    'Tokeniser': tokeniser,
                    'Fertility_Penalty': fertility,
                    'Is_Problematic': fertility > MAX_TOKEN_FERTILITY,
                    'Severity': severity
                })
        
        return pd.DataFrame(rows)
    
    def analyse_subword_piecing(self, 
                               sample_words: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyse subword piecing patterns"""
        results = []
        
        for lang, words in sample_words.items():
            if lang == 'English' or not words:
                continue
            
            total_chars = 0
            total_subwords = 0
            
            for word in words[:100]:
                total_chars += len(word)
                subwords = self._simulate_subword_split(word)
                total_subwords += len(subwords)
            
            avg_pieces_per_char = total_subwords / max(total_chars, 1)
            piecing_ratio = avg_pieces_per_char * 10
            
            if piecing_ratio > 0.5:
                severity = 'Critical'
            elif piecing_ratio > SENTENCE_PIECING_THRESHOLD:
                severity = 'High'
            elif piecing_ratio > 0.15:
                severity = 'Moderate'
            else:
                severity = 'Low'
            
            results.append({
                'Language': lang,
                'Sample_Words_Analyzed': min(100, len(words)),
                'Avg_Subword_Pieces_Per_Word': total_subwords / max(min(100, len(words)), 1),
                'Piecing_Ratio': piecing_ratio,
                'Severity': severity,
                'Is_Highly_Pieced': piecing_ratio > SENTENCE_PIECING_THRESHOLD
            })
        
        return pd.DataFrame(results)
    
    def _simulate_subword_split(self, word: str) -> List[str]:
        """Simulate subword splitting for a word with BPE-like behavior"""
        if len(word) <= 4:
            return [word]
        
        # Common prefixes and suffixes in East African languages
        prefixes = ['mu', 'ku', 'ni', 'tu', 'ba', 'wa', 'ki', 'vi', 'a', 'e', 'i', 'o', 'u']
        suffixes = ['a', 'e', 'i', 'o', 'u', 'ka', 'ta', 'na', 'za', 'ya', 'ni', 'ku']
        
        result = []
        remaining = word
        
        # Check for prefixes
        for pref in prefixes:
            if remaining.startswith(pref) and len(pref) < len(remaining) - 1:
                result.append(pref)
                remaining = remaining[len(pref):]
                break
        
        # Split the rest
        while len(remaining) > 4:
            found = False
            # Check for suffixes
            for suff in suffixes:
                if remaining.endswith(suff) and len(remaining) - len(suff) >= 3:
                    result.append(remaining[:-len(suff)])
                    result.append(suff)
                    remaining = ""
                    found = True
                    break
            
            if not found:
                # Split at a natural boundary
                split_point = min(3, len(remaining) - 2)
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
            if lang not in normalized_texts or not normalized_texts[lang]:
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
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in answers_by_lang or not answers_by_lang[lang]:
                continue
            
            answers = answers_by_lang[lang]
            
            # Answer diversity
            answer_starts = [a[:30] if len(a) > 30 else a for a in answers]
            unique_starts = len(set(answer_starts))
            diversity_ratio = unique_starts / max(len(answers), 1)
            
            # Semantic consistency
            sentences = []
            for a in answers:
                sentences.extend(a.split('.'))
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            
            avg_sent_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
            
            results.append({
                'Language': lang,
                'Answer_Diversity_Ratio': diversity_ratio,
                'Total_Answers': len(answers),
                'Avg_Sentence_Length_Trust': avg_sent_len
            })
        
        # Compare to English
        english_diversity = next((r['Answer_Diversity_Ratio'] for r in results if r['Language'] == 'English'), 0)
        
        trust_aware_results = []
        for r in results:
            if r['Language'] != 'English' and english_diversity > 0:
                diversity_ratio = r['Answer_Diversity_Ratio'] / english_diversity
                if diversity_ratio > 0.8:
                    trust_level = 'High'
                    trust_color = 'green'
                elif diversity_ratio > 0.6:
                    trust_level = 'Medium'
                    trust_color = 'orange'
                else:
                    trust_level = 'Low'
                    trust_color = 'red'
                
                trust_aware_results.append({
                    'Comparison': f"{r['Language']} vs English",
                    'Diversity_Ratio': round(diversity_ratio, 3),
                    'Trust_Level': trust_level,
                    'Trust_Color': trust_color,
                    'Interpretation': f"{r['Language']} responses show {diversity_ratio:.1%} answer diversity compared to English"
                })
        
        return trust_aware_results
    
    def analyse_content_bias(self, 
                            normalized_texts: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyse content bias across languages using domain keywords"""
        results = []
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in normalized_texts or not normalized_texts[lang]:
                continue
            
            texts = normalized_texts[lang]
            lang_keywords = DOMAIN_KEYWORDS.get(lang, DOMAIN_KEYWORDS.get('English', {}))
            
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
                            piecing_df: pd.DataFrame,
                            complexity_df: pd.DataFrame,
                            trust_metrics: List[Dict]) -> List[Dict]:
        """Generate flags for linguistic biases with enhanced severity"""
        flags = []
        
        # Tokeniser flags
        if not tokeniser_df.empty:
            for _, row in tokeniser_df.iterrows():
                if row.get('Severity') in ['Critical', 'High']:
                    flags.append({
                        'Type': 'Tokenisation_Bias',
                        'Language': row['Language'],
                        'Tokeniser': row['Tokeniser'],
                        'Severity': row['Severity'],
                        'Description': f"High fertility penalty ({row['Fertility_Penalty']:.2f}) for {row['Tokeniser']}",
                        'Recommendation': 'Consider using a different tokeniser for this language'
                    })
        
        # Subword piecing flags
        if not piecing_df.empty:
            for _, row in piecing_df.iterrows():
                if row.get('Severity') in ['Critical', 'High']:
                    flags.append({
                        'Type': 'Subword_Piecing',
                        'Language': row['Language'],
                        'Severity': row['Severity'],
                        'Description': f"High subword piecing ratio ({row['Piecing_Ratio']:.2f})",
                        'Recommendation': 'Review if over-tokenisation is affecting semantic understanding'
                    })
        
        # Structural complexity flags
        if not complexity_df.empty:
            english_complexity = complexity_df[complexity_df['Language'] == 'English']['Structural_Complexity_Mean'].values
            if len(english_complexity) > 0:
                english_complexity = english_complexity[0]
                for _, row in complexity_df.iterrows():
                    if row['Language'] != 'English':
                        ratio = row['Structural_Complexity_Mean'] / max(english_complexity, 0.001)
                        if ratio < 0.4:
                            severity = 'Critical'
                        elif ratio < 0.6:
                            severity = 'High'
                        elif ratio < 0.8:
                            severity = 'Moderate'
                        else:
                            severity = 'Low'
                        
                        if severity in ['Critical', 'High']:
                            flags.append({
                                'Type': 'Structural_Bias',
                                'Language': row['Language'],
                                'Severity': severity,
                                'Description': f"Structural complexity is {ratio:.1%} of English",
                                'Recommendation': 'Review if simpler responses indicate content loss'
                            })
        
        # Trust metrics flags
        for metric in trust_metrics:
            if metric['Trust_Level'] == 'Low':
                flags.append({
                    'Type': 'Trust_Bias_Critical',
                    'Comparison': metric['Comparison'],
                    'Severity': 'Critical',
                    'Description': metric['Interpretation'],
                    'Recommendation': 'URGENT: Review translation quality and cultural adaptation'
                })
            elif metric['Trust_Level'] == 'Medium':
                flags.append({
                    'Type': 'Trust_Bias',
                    'Comparison': metric['Comparison'],
                    'Severity': 'Moderate',
                    'Description': metric['Interpretation'],
                    'Recommendation': 'Monitor translation quality'
                })
        
        return flags
    def run_full_audit(self,
                  questions_by_lang: Dict[str, List[str]],
                  answers_by_lang: Dict[str, List[str]],
                  tokeniser_perfs: Dict[str, Dict[str, float]],
                  sample_words: Dict[str, List[str]]) -> Dict:
        """Run complete linguistic bias audit"""
        self.logger.info("="*50)
        self.logger.info("STARTING LINGUISTIC BIAS AUDIT")
        self.logger.info("="*50)
        
        self.logger.info("Analysing tokeniser performance...")
        tokeniser_df = self.analyse_tokeniser_performance(tokeniser_perfs)
        
        self.logger.info("Analysing subword piecing...")
        piecing_df = self.analyse_subword_piecing(sample_words)
        
        self.logger.info("Analysing structural complexity...")
        complexity_df = self.analyse_structural_complexity(answers_by_lang)
        
        self.logger.info("Computing trust-aware metrics...")
        trust_metrics = self.compute_trust_aware_metrics(questions_by_lang, answers_by_lang)
        
        self.logger.info("Analysing content bias...")
        content_df = self.analyse_content_bias(answers_by_lang)
        
        self.logger.info("Detecting linguistic flags...")
        flags = self.detect_linguistic_flags(tokeniser_df, piecing_df, complexity_df, trust_metrics)
        
        # Fix: Use proper pandas column access
        problematic_count = 0
        critical_tokenisers = 0
        if not tokeniser_df.empty and 'Severity' in tokeniser_df.columns:
            problematic_count = len(tokeniser_df[tokeniser_df['Severity'].isin(['Critical', 'High'])])
            critical_tokenisers = len(tokeniser_df[tokeniser_df['Severity'] == 'Critical'])
        
        high_piecing_count = 0
        if not piecing_df.empty and 'Severity' in piecing_df.columns:
            high_piecing_count = len(piecing_df[piecing_df['Severity'].isin(['Critical', 'High'])])
        
        summary = {
            'tokenisers_analyzed': len(tokeniser_df['Tokeniser'].unique()) if not tokeniser_df.empty else 0,
            'problematic_tokenisers': problematic_count,
            'critical_tokenisers': critical_tokenisers,
            'high_piecing_languages': high_piecing_count,
            'trust_issues': len([m for m in trust_metrics if m['Trust_Level'] == 'Low']),
            'flags_generated': len(flags),
            'critical_flags': sum(1 for f in flags if f.get('Severity') == 'Critical')
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
        
        self.logger.info("="*50)
        self.logger.info("LINGUISTIC AUDIT COMPLETE")
        self.logger.info(f"  Tokenisers analyzed: {summary['tokenisers_analyzed']}")
        self.logger.info(f"  Problematic tokenisers: {summary['problematic_tokenisers']}")
        self.logger.info(f"  Critical flags: {summary['critical_flags']}")
        self.logger.info("="*50)
        
        return results