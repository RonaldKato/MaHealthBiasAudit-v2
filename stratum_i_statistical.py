"""
Stratum I: Statistical Bias Audit
Based on English, Swahili, Luganda, Runyankore
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter
from dataclasses import dataclass
from scipy.stats import ks_2samp, entropy
import warnings
warnings.filterwarnings('ignore')

from config import THRESHOLDS, PRIMARY_LANGUAGES
from utils import (
    compute_jensen_shannon_divergence, compute_hapax_legomena,
    set_seed, RANDOM_SEED
)


@dataclass
class CorpusStats:
    language: str
    vocab_size: int
    type_token_ratio: float
    hapax_proportion: float
    avg_q_len: float
    avg_a_len: float


@dataclass
class ISDResult:
    language_pair: Tuple[str, str]
    isd_value: float
    needs_intervention: bool


class StatisticalBiasAuditor:
    """Stratum I: Statistical bias audit"""
    
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.corpus_stats = {}
        self.freq_dists = {}
        self.isd_results = []
        self.jsd_results = {}
    
    def compute_corpus_statistics(self, questions: Dict[str, List[str]], 
                                   answers: Dict[str, List[str]]) -> Dict[str, CorpusStats]:
        """Compute per-language corpus statistics"""
        print("\n" + "="*60)
        print("Corpus Statistics (Data)")
        print("="*60)
        
        for lang in PRIMARY_LANGUAGES:
            q_texts = questions.get(lang, [])
            a_texts = answers.get(lang, [])
            
            all_text = ' '.join(q_texts) + ' ' + ' '.join(a_texts)
            tokens = all_text.lower().split()
            
            vocab = set(tokens)
            ttr = len(vocab) / max(len(tokens), 1)
            _, hapax = compute_hapax_legomena(tokens)
            
            q_lens = [len(q.split()) for q in q_texts if q]
            a_lens = [len(a.split()) for a in a_texts if a]
            
            stats = CorpusStats(
                language=lang,
                vocab_size=len(vocab),
                type_token_ratio=ttr,
                hapax_proportion=hapax,
                avg_q_len=np.mean(q_lens) if q_lens else 0,
                avg_a_len=np.mean(a_lens) if a_lens else 0
            )
            self.corpus_stats[lang] = stats
            
            print(f"   {lang}: TTR={ttr:.3f}, Vocab={len(vocab)}, Hapax={hapax:.3f}")
        
        return self.corpus_stats
    
    def compute_frequency_distributions(self, texts: Dict[str, List[str]]) -> Dict[str, Counter]:
        """Compute token frequency distributions"""
        print("\n" + "="*60)
        print("Frequency Distributions")
        print("="*60)
        
        for lang, text_list in texts.items():
            all_tokens = []
            for text in text_list:
                all_tokens.extend(text.lower().split())
            self.freq_dists[lang] = Counter(all_tokens)
            print(f"   {lang}: {len(self.freq_dists[lang])} unique tokens")
        
        return self.freq_dists
    
    def compute_information_specific_divergence(self) -> List[ISDResult]:
        """Compute ISD between language pairs"""
        print("\n" + "="*60)
        print("Information Specific Divergence (ISD)")
        print("="*60)
        
        self.isd_results = []
        
        for i, lang1 in enumerate(PRIMARY_LANGUAGES):
            for j, lang2 in enumerate(PRIMARY_LANGUAGES):
                if i >= j:
                    continue
                
                dist1 = self.freq_dists[lang1]
                dist2 = self.freq_dists[lang2]
                
                total1 = sum(dist1.values())
                total2 = sum(dist2.values())
                all_tokens = set(dist1.keys()) | set(dist2.keys())
                
                epsilon = 1e-12
                p1 = np.array([(dist1.get(tok, 0) + epsilon) / (total1 + epsilon * len(all_tokens)) 
                              for tok in all_tokens])
                p2 = np.array([(dist2.get(tok, 0) + epsilon) / (total2 + epsilon * len(all_tokens)) 
                              for tok in all_tokens])
                
                p1 = p1 / np.sum(p1)
                p2 = p2 / np.sum(p2)
                
                isd = entropy(p1, p2) + entropy(p2, p1)
                needs_intervention = isd > THRESHOLDS['jsd_high']
                
                result = ISDResult(
                    language_pair=(lang1, lang2),
                    isd_value=isd,
                    needs_intervention=needs_intervention
                )
                self.isd_results.append(result)
                
                status = "⚠️" if needs_intervention else "✓"
                print(f"   {lang1} ↔ {lang2}: ISD={isd:.4f} {status}")
        
        return self.isd_results
    
    def compute_jensen_shannon_divergence(self, questions: Dict[str, List[str]]) -> Dict:
        """Compute JSD between language pairs"""
        print("\n" + "="*60)
        print("Jensen-Shannon Divergence")
        print("="*60)
        
        unigrams = {}
        for lang in PRIMARY_LANGUAGES:
            all_tokens = []
            for text in questions.get(lang, []):
                all_tokens.extend(text.lower().split())
            
            if all_tokens:
                counter = Counter(all_tokens)
                total = sum(counter.values())
                unigrams[lang] = {tok: cnt/total for tok, cnt in counter.items()}
        
        for i, lang1 in enumerate(PRIMARY_LANGUAGES):
            for j, lang2 in enumerate(PRIMARY_LANGUAGES):
                if i < j and lang1 in unigrams and lang2 in unigrams:
                    jsd = compute_jensen_shannon_divergence(unigrams[lang1], unigrams[lang2])
                    self.jsd_results[f"{lang1}_vs_{lang2}"] = jsd
                    print(f"   {lang1} ↔ {lang2}: JSD={jsd:.4f}")
        
        return self.jsd_results
    
    def compute_length_distribution_test(self, questions: Dict[str, List[str]]) -> Dict:
        """K-S test for length distributions"""
        print("\n" + "="*60)
        print("Length Distribution Test")
        print("="*60)
        
        ks_results = {}
        
        for i, lang1 in enumerate(PRIMARY_LANGUAGES):
            for j, lang2 in enumerate(PRIMARY_LANGUAGES):
                if i < j:
                    len1 = [len(q.split()) for q in questions.get(lang1, []) if q]
                    len2 = [len(q.split()) for q in questions.get(lang2, []) if q]
                    
                    if len1 and len2:
                        ks_stat, p_value = ks_2samp(len1, len2)
                        ks_results[f"{lang1}_vs_{lang2}"] = {
                            'ks_stat': ks_stat,
                            'p_value': p_value,
                            'different': p_value < 0.05
                        }
                        status = "DIFFERENT" if p_value < 0.05 else "SIMILAR"
                        print(f"   {lang1} vs {lang2}: KS={ks_stat:.3f}, p={p_value:.4f} [{status}]")
        
        return ks_results
    
    def get_flags(self) -> List[str]:
        """Generate flags based on statistical bias"""
        flags = []
        for result in self.isd_results:
            if result.needs_intervention:
                flags.append(f"HIGH_ISD: {result.language_pair[0]}-{result.language_pair[1]} = {result.isd_value:.3f}")
        return flags
    
    def run_full_audit(self, questions: Dict[str, List[str]], 
                        answers: Dict[str, List[str]]) -> Dict:
        """Run statistical audit"""
        print("\n" + "="*70)
        print("STRATUM I: Statistical Bias Audit")
        print("="*70)
        
        corpus = self.compute_corpus_statistics(questions, answers)
        freq = self.compute_frequency_distributions(answers)
        isd = self.compute_information_specific_divergence()
        jsd = self.compute_jensen_shannon_divergence(questions)
        ks = self.compute_length_distribution_test(questions)
        
        return {
            'corpus_statistics': corpus,
            'isd_results': isd,
            'jsd_results': jsd,
            'ks_test_results': ks,
            'flags': self.get_flags()
        }