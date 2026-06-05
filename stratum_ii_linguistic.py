"""
Stratum II: Linguistic Bias Audit
Based on English, Swahili, Luganda, Runyankore
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter

from config import THRESHOLDS, PRIMARY_LANGUAGES, INTERROGATIVE_PATTERNS, CULTURAL_TERMINOLOGY
from utils import set_seed, RANDOM_SEED, extract_cultural_terms


@dataclass
class TrustResult:
    language: str
    cultural_terms_found: List[Dict]
    trust_score: float
    preservation_needed: bool


class LinguisticBiasAuditor:
    """Stratum II: Linguistic bias audit"""
    
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.tokenisation_results = None
        self.morph_results = {}
        self.trust_results = {}
    
    def compute_tokenisation_parity(self, tokeniser_perfs: Dict[str, Dict]) -> pd.DataFrame:
        """Compute tokenisation parity"""
        print("\n" + "="*60)
        print("Tokenisation Parity")
        print("="*60)
        
        rows = []
        for lang in PRIMARY_LANGUAGES:
            if lang == 'English':
                continue
            
            for tokeniser in ['mBERT', 'XLM-R', 'AfriBERTa']:
                fertility = tokeniser_perfs.get(lang, {}).get(tokeniser, 1.0)
                rows.append({
                    'Language': lang,
                    'Tokeniser': tokeniser,
                    'Fertility_Penalty': fertility,
                    'TP_Flag': fertility > THRESHOLDS['tokenisation_parity']
                })
        
        df = pd.DataFrame(rows)
        self.tokenisation_results = df
        
        for lang in PRIMARY_LANGUAGES:
            if lang != 'English':
                lang_df = df[df['Language'] == lang]
                if not lang_df.empty:
                    avg_tp = lang_df['Fertility_Penalty'].mean()
                    status = "⚠️" if avg_tp > THRESHOLDS['tokenisation_parity'] else "✓"
                    print(f"   {lang}: avg TP={avg_tp:.2f} {status}")
        
        return df
    
    def analyze_interrogative_structure(self, questions: Dict[str, List[str]]) -> pd.DataFrame:
        """Analyze interrogative structure patterns"""
        print("\n" + "="*60)
        print("❓ Interrogative Structure Analysis")
        print("="*60)
        
        rows = []
        
        for lang in PRIMARY_LANGUAGES:
            pattern_info = INTERROGATIVE_PATTERNS.get(lang, INTERROGATIVE_PATTERNS['English'])
            expected_type = pattern_info['type']
            
            for q in questions.get(lang, []):
                q_lower = q.lower()
                
                # Detect interrogative position
                if q_lower.startswith(('what', 'when', 'where', 'why', 'how', 'which', 'who', 'have', 'do', 'are', 'is', 'can')):
                    actual_type = 'wh_fronted'
                elif any(marker in q_lower for marker in ['je', 'nini', 'gani']):
                    actual_type = 'mixed'
                elif q_lower.endswith('?'):
                    actual_type = 'sentence_final'
                else:
                    actual_type = 'verb_internal'
                
                is_mismatch = expected_type != actual_type
                
                rows.append({
                    'Language': lang,
                    'Question': q[:60] + '...' if len(q) > 60 else q,
                    'Expected_Type': expected_type,
                    'Actual_Type': actual_type,
                    'Mismatch': is_mismatch
                })
        
        df = pd.DataFrame(rows)
        
        print(f"\n   Results:")
        for lang in PRIMARY_LANGUAGES:
            lang_df = df[df['Language'] == lang]
            mismatches = lang_df['Mismatch'].sum()
            total = len(lang_df)
            print(f"   {lang}: {mismatches}/{total} mismatches ({mismatches/total*100:.0f}%)")
        
        return df
    
    def trust_aware_module(self, texts: Dict[str, List[str]]) -> Dict[str, TrustResult]:
        """Trust-aware analysis for cultural terminology"""
        print("\n" + "="*60)
        print("🤝 Trust-Aware Module")
        print("="*60)
        
        results = {}
        
        for lang in PRIMARY_LANGUAGES:
            all_text = ' '.join(texts.get(lang, []))
            cultural_terms = extract_cultural_terms(all_text, lang)
            
            if cultural_terms:
                trust_score = min(1.0, 0.5 + sum(t['cultural_importance'] for t in cultural_terms) / len(cultural_terms) * 0.3)
                preservation_needed = any(t['is_medical'] for t in cultural_terms)
            else:
                trust_score = 0.5
                preservation_needed = False
            
            results[lang] = TrustResult(
                language=lang,
                cultural_terms_found=cultural_terms,
                trust_score=trust_score,
                preservation_needed=preservation_needed
            )
            
            status = "PRESERVE" if preservation_needed else "REVIEW"
            print(f"   {lang}: Trust={trust_score:.2f}, Terms={len(cultural_terms)} [{status}]")
        
        self.trust_results = results
        return results
    
    def compute_morphological_alignment(self, sample_words: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """Compute morphological alignment scores"""
        print("\n" + "="*60)
        print("Morphological Alignment")
        print("="*60)
        
        results = {}
        
        for lang in PRIMARY_LANGUAGES:
            words = sample_words.get(lang, [])
            lang_results = []
            
            for word in words:
                # Simulated morpheme splitting
                if len(word) > 6 and lang in ['Luganda', 'Runyankore']:
                    splits = [word[:3], word[3:6], word[6:]] if len(word) > 9 else [word[:3], word[3:]]
                elif len(word) > 6 and lang == 'Swahili':
                    splits = [word[:4], word[4:]] if len(word) > 8 else [word]
                else:
                    splits = [word]
                
                lang_results.append({
                    'word': word,
                    'segments': '|'.join(splits),
                    'boundary_f1': 0.85 if len(splits) > 1 else 1.0,
                    'is_aligned': len(splits) == 1
                })
            
            results[lang] = pd.DataFrame(lang_results)
            if not results[lang].empty:
                avg_f1 = results[lang]['boundary_f1'].mean()
                print(f"   {lang}: MAS={avg_f1:.3f}")
        
        return results
    
    def get_flags(self) -> List[str]:
        """Generate linguistic bias flags"""
        flags = []
        
        if self.tokenisation_results is not None:
            high_tp = self.tokenisation_results[self.tokenisation_results['TP_Flag']]
            for _, row in high_tp.iterrows():
                flags.append(f"HIGH_FERTILITY: {row['Language']} TP={row['Fertility_Penalty']:.2f}")
        
        for lang, result in self.trust_results.items():
            if result.preservation_needed:
                flags.append(f"PRESERVE_CULTURAL: {lang} has {len(result.cultural_terms_found)} cultural terms")
        
        return flags
    
    def run_full_audit(self, questions: Dict[str, List[str]], 
                        answers: Dict[str, List[str]],
                        tokeniser_perfs: Dict[str, Dict],
                        sample_words: Dict[str, List[str]]) -> Dict:
        """Run linguistic audit"""
        print("\n" + "="*70)
        print("STRATUM II: Linguistic Bias Audit")
        print("="*70)
        
        tp_df = self.compute_tokenisation_parity(tokeniser_perfs)
        inter_df = self.analyze_interrogative_structure(questions)
        trust_results = self.trust_aware_module(answers)
        morph_results = self.compute_morphological_alignment(sample_words)
        
        return {
            'tokenisation_parity': tp_df,
            'interrogative_analysis': inter_df,
            'trust_aware_results': trust_results,
            'morphological_alignment': morph_results,
            'flags': self.get_flags()
        }