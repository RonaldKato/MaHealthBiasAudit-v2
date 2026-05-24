"""
Stratum II: Linguistic and Structural Bias Audit
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter

from config import (
    LANGUAGES, PRIMARY_LANGUAGES, THRESHOLDS, 
    INTERROGATIVE_PATTERNS, CULTURAL_TERMINOLOGY, DIALECT_MARKERS
)
from utils import (
    normalize_text, compute_oov_rate,
    get_morpheme_boundaries, compute_boundary_f1, set_seed, RANDOM_SEED
)


@dataclass
class TrustAwareResult:
    """Trust-aware linguistic analysis result"""
    language: str
    cultural_terms_found: List[Dict]
    trust_score: float
    recommendations: List[str]
    preservation_needed: bool


class LinguisticBiasAuditor:
    """
    Stratum II: Linguistic and structural bias analysis
    Includes tokenisation parity, morphological analysis,
    interrogative structure analysis, and trust-aware module
    """
    
    def __init__(self):
        """Initialize linguistic bias auditor"""
        set_seed(RANDOM_SEED)
        self.tokenisation_results = pd.DataFrame()  # Initialize as empty DataFrame
        self.morphological_results: Dict[str, pd.DataFrame] = {}
        self.trust_results: Dict[str, TrustAwareResult] = {}
        self.interrogative_analysis = pd.DataFrame()  # Initialize as empty DataFrame
    
    def compute_tokenisation_parity(self, tokeniser_performances: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compute Tokenisation Parity (TP) across languages
        TP(L) = mean_s [token_count(s_L) / token_count(s_en)]
        """
        print("\n" + "="*60)
        print(" Computing Tokenisation Parity")
        print("="*60)
        
        rows = []
        
        for lang, perf in tokeniser_performances.items():
            if lang == 'English':
                continue
            
            for tokeniser in ['mBERT', 'XLM-R', 'AfriBERTa']:
                fertility = perf.get(tokeniser, {}).get('fertility', 1.0)
                
                rows.append({
                    'Language': lang,
                    'Tokeniser': tokeniser,
                    'Fertility_Penalty': fertility,
                    'TP_Flag': fertility > THRESHOLDS['tokenisation_parity'],
                    'Excess_Tokens_Pct': (fertility - 1) * 100
                })
        
        self.tokenisation_results = pd.DataFrame(rows)
        
        # Print summary only if DataFrame is not empty
        if not self.tokenisation_results.empty:
            print(f"\n   Tokenisation Parity Summary:")
            for lang in self.tokenisation_results['Language'].unique():
                lang_data = self.tokenisation_results[self.tokenisation_results['Language'] == lang]
                avg_fertility = lang_data['Fertility_Penalty'].mean()
                status = "HIGH" if avg_fertility > THRESHOLDS['tokenisation_parity'] else "✓ OK"
                print(f"   {lang}: avg TP={avg_fertility:.2f} [{status}]")
        
        return self.tokenisation_results
    
    def compute_morphological_alignment(self, 
                                         sample_words: Dict[str, List[str]],
                                         tokeniser_segmentations: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Compute Morphological Alignment Score (MAS)
        """
        print("\n" + "="*60)
        print(" Computing Morphological Alignment")
        print("="*60)
        
        results = {}
        
        for lang, words in sample_words.items():
            lang_results = []
            
            for word in words:
                # Get tokeniser segmentation
                tokeniser_segs = tokeniser_segmentations.get(lang, {}).get(word, [word])
                
                # Get morpheme segmentation
                morpheme_segs = self._get_morpheme_segmentation(word, lang)
                
                # Compute boundary F1
                token_boundaries = get_morpheme_boundaries(word, tokeniser_segs)
                morpheme_boundaries = get_morpheme_boundaries(word, morpheme_segs)
                boundary_f1 = compute_boundary_f1(token_boundaries, morpheme_boundaries)
                
                # Fragmentation score
                fragmentation = len(tokeniser_segs) / max(len(morpheme_segs), 1)
                
                lang_results.append({
                    'word': word,
                    'tokeniser_segments': '|'.join(tokeniser_segs),
                    'morpheme_segments': '|'.join(morpheme_segs),
                    'boundary_f1': boundary_f1,
                    'fragmentation_score': fragmentation,
                    'is_aligned': boundary_f1 > THRESHOLDS['mas_threshold']
                })
            
            results[lang] = pd.DataFrame(lang_results)
            self.morphological_results[lang] = results[lang]
            
            if not results[lang].empty:
                avg_f1 = results[lang]['boundary_f1'].mean()
                status = "✓ GOOD" if avg_f1 > THRESHOLDS['mas_threshold'] else " NEEDS IMPROVEMENT"
                print(f"   {lang}: MAS={avg_f1:.3f} [{status}]")
        
        return results
    
    def _get_morpheme_segmentation(self, word: str, lang: str) -> List[str]:
        """Get morpheme segmentation (simulated)"""
        # Simple heuristic segmentation
        if len(word) > 6 and lang in ['Luganda', 'Runyankore', 'Swahili']:
            # Split into prefix, stem, suffix
            mid = len(word) // 3
            return [word[:mid], word[mid:2*mid], word[2*mid:]] if len(word) > 9 else [word[:3], word[3:]]
        return [word]
    
    def analyze_interrogative_structure(self, questions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Analyze interrogative structure patterns across languages
        """
        print("\n" + "="*60)
        print("❓ Analyzing Interrogative Structure")
        print("="*60)
        
        rows = []
        mismatch_count = 0
        
        for lang, lang_questions in questions.items():
            pattern_info = INTERROGATIVE_PATTERNS.get(lang, INTERROGATIVE_PATTERNS['English'])
            pattern = pattern_info['pattern']
            expected_type = pattern_info['type']
            
            for q in lang_questions:
                # Detect interrogative position
                match = re.search(pattern, q.lower()) if isinstance(pattern, str) else None
                interrogative_position = match.start() if match else -1
                
                # Determine actual type based on position
                if interrogative_position == 0:
                    actual_type = 'wh_fronted'
                elif interrogative_position > 0:
                    actual_type = 'inline'
                else:
                    actual_type = 'sentence_final'
                
                is_mismatch = expected_type != actual_type
                if is_mismatch:
                    mismatch_count += 1
                
                rows.append({
                    'Language': lang,
                    'Question': q[:80] + '...' if len(q) > 80 else q,
                    'Expected_Type': expected_type,
                    'Actual_Type': actual_type,
                    'Interrogative_Position': interrogative_position,
                    'Mismatch': is_mismatch,
                    'Recommendation': 'Adapt QA model for non-English patterns' if is_mismatch else None
                })
        
        self.interrogative_analysis = pd.DataFrame(rows)
        
        print(f"\n   Interrogative Structure Summary:")
        if not self.interrogative_analysis.empty:
            for lang in self.interrogative_analysis['Language'].unique():
                lang_df = self.interrogative_analysis[self.interrogative_analysis['Language'] == lang]
                mismatches = lang_df['Mismatch'].sum()
                total = len(lang_df)
                print(f"   {lang}: {mismatches}/{total} mismatches ({mismatches/total*100:.1f}%)")
        
        return self.interrogative_analysis
    
    def trust_aware_module(self, texts: List[str], language: str) -> TrustAwareResult:
        """
        Trust-Aware Module for cultural terminology and phrases
        """
        print(f"\n   Trust-Aware Analysis for {language}...")
        
        cultural_terms_found = []
        text_lower = ' '.join(texts).lower()
        
        # Search for cultural terminology
        if language in CULTURAL_TERMINOLOGY:
            for term, (category, translation, importance, is_medical) in CULTURAL_TERMINOLOGY[language].items():
                if term.lower() in text_lower:
                    cultural_terms_found.append({
                        'term': term,
                        'category': category,
                        'translation': translation,
                        'cultural_importance': importance,
                        'is_medical': is_medical
                    })
        
        # Compute trust score
        if cultural_terms_found:
            cultural_bonus = sum(t['cultural_importance'] for t in cultural_terms_found) / len(cultural_terms_found)
            trust_score = min(1.0, 0.5 + cultural_bonus * 0.5)
            preservation_needed = any(t['is_medical'] for t in cultural_terms_found)
        else:
            trust_score = 0.6  # Neutral baseline
            preservation_needed = False
        
        # Generate recommendations
        recommendations = []
        if not cultural_terms_found:
            recommendations.append(f"Add culturally specific {language} medical terminology to improve trust")
        for term in cultural_terms_found:
            if term['is_medical']:
                recommendations.append(f"✓ PRESERVE: '{term['term']}' ({term['translation']}) is valid cultural medical knowledge")
            else:
                recommendations.append(f"Document: '{term['term']}' has cultural significance")
        
        result = TrustAwareResult(
            language=language,
            cultural_terms_found=cultural_terms_found,
            trust_score=trust_score,
            recommendations=recommendations,
            preservation_needed=preservation_needed
        )
        
        self.trust_results[language] = result
        
        # Print summary
        status = "✓ PRESERVE" if preservation_needed else "REVIEW"
        print(f"      {status}: Trust Score={trust_score:.2f}, "
              f"Terms found={len(cultural_terms_found)}")
        
        return result
    
    def detect_dialect_variance(self, texts: List[str], language: str) -> Dict:
        """
        Detect dialect variance during tokenisation and morphology
        """
        if language not in DIALECT_MARKERS:
            return {'variance_detected': False, 'primary_dialect': None}
        
        markers = DIALECT_MARKERS[language]
        text_lower = ' '.join(texts).lower()
        
        dialect_scores = {}
        for dialect, dialect_markers in markers.items():
            score = 0
            for marker in dialect_markers:
                if marker.lower() in text_lower:
                    score += 1
            dialect_scores[dialect] = score / max(len(dialect_markers), 1)
        
        if dialect_scores:
            primary = max(dialect_scores, key=dialect_scores.get)
            variance_detected = max(dialect_scores.values()) > 0.25
        else:
            primary = None
            variance_detected = False
        
        return {
            'language': language,
            'dialect_scores': dialect_scores,
            'variance_detected': variance_detected,
            'primary_dialect': primary,
            'tokenisation_impact': 'high' if variance_detected else 'low',
            'recommendations': [
                f"Consider {primary}-specific tokenisation for better accuracy" if primary and variance_detected else None
            ]
        }
    
    def compute_morphological_alignment_scores(self) -> pd.DataFrame:
        """Create summary DataFrame of morphological alignment scores"""
        rows = []
        
        for lang, df in self.morphological_results.items():
            if df is not None and not df.empty:
                rows.append({
                    'Language': lang,
                    'Avg_Boundary_F1': round(df['boundary_f1'].mean(), 4),
                    'Avg_Fragmentation': round(df['fragmentation_score'].mean(), 4),
                    'Pct_Aligned': round((df['is_aligned'].sum() / len(df)) * 100, 1),
                    'MAS_Flag': df['boundary_f1'].mean() < THRESHOLDS['mas_threshold'],
                    'Severity': 'High' if df['boundary_f1'].mean() < 0.5 else 
                               'Moderate' if df['boundary_f1'].mean() < THRESHOLDS['mas_threshold'] else 'Low'
                })
        
        return pd.DataFrame(rows)
    
    def get_flags(self) -> List[str]:
        """Generate flags based on linguistic bias thresholds"""
        flags = []
        
        # Fix: Check if DataFrame is not empty using .empty attribute (NOT boolean evaluation)
        if self.tokenisation_results is not None and not self.tokenisation_results.empty:
            high_tp = self.tokenisation_results[self.tokenisation_results['TP_Flag']]
            for _, row in high_tp.iterrows():
                flags.append(f"HIGH_FERTILITY: {row['Language']} with {row['Tokeniser']} "
                           f"(TP={row['Fertility_Penalty']:.2f}, excess={row['Excess_Tokens_Pct']:.0f}%)")
        
        # Fix: Check morphological alignment scores
        mas_df = self.compute_morphological_alignment_scores()
        if mas_df is not None and not mas_df.empty:
            low_mas = mas_df[mas_df['MAS_Flag']]
            for _, row in low_mas.iterrows():
                flags.append(f"POOR_MORPH_ALIGNMENT: {row['Language']} "
                           f"(MAS={row['Avg_Boundary_F1']:.3f}, aligned={row['Pct_Aligned']:.1f}%)")
        
        # Fix: Check interrogative analysis
        if self.interrogative_analysis is not None and not self.interrogative_analysis.empty:
            mismatches = self.interrogative_analysis[self.interrogative_analysis['Mismatch']]
            if not mismatches.empty:
                by_lang = mismatches.groupby('Language').size().to_dict()
                flags.append(f"QUERY_STRUCTURE_MISMATCH: {len(mismatches)} questions across {len(by_lang)} languages")
        
        # Trust-aware flags
        for lang, result in self.trust_results.items():
            if result.preservation_needed:
                flags.append(f"PRESERVE_CULTURAL: {lang} - {len(result.cultural_terms_found)} medical terms to preserve")
            elif result.trust_score < THRESHOLDS['trust_score_target']:
                flags.append(f"LOW_TRUST: {lang} (score={result.trust_score:.2f}) - needs cultural adaptation")
        
        return flags
    
    def run_full_audit(self, questions: Dict[str, List[str]], 
                        answers: Dict[str, List[str]],
                        tokeniser_performances: Dict[str, Dict],
                        sample_words: Dict[str, List[str]],
                        tokeniser_segmentations: Dict[str, Dict]) -> Dict:
        """
        Run complete linguistic bias audit
        """
        print("\n" + "="*70)
        print("STRATUM II: Linguistic Bias Audit")
        print("="*70)
        
        # Compute tokenisation parity
        tp_df = self.compute_tokenisation_parity(tokeniser_performances)
        
        # Compute morphological alignment
        morph_results = self.compute_morphological_alignment(sample_words, tokeniser_segmentations)
        
        # Analyze interrogative structure
        inter_df = self.analyze_interrogative_structure(questions)
        
        # Trust-aware module for each language
        trust_results = {}
        for lang, text_list in answers.items():
            trust_results[lang] = self.trust_aware_module(text_list, lang)
        
        # Detect dialect variance
        dialect_results = {}
        for lang, text_list in answers.items():
            dialect_results[lang] = self.detect_dialect_variance(text_list, lang)
        
        # Compute morphological alignment scores summary
        mas_df = self.compute_morphological_alignment_scores()
        
        # Generate flags
        flags = self.get_flags()
        
        return {
            'tokenisation_parity': tp_df,
            'morphological_alignment': morph_results,
            'morphological_alignment_summary': mas_df,
            'interrogative_analysis': inter_df,
            'trust_aware_results': trust_results,
            'dialect_variance': dialect_results,
            'flags': flags
        }


# Test the auditor
if __name__ == "__main__":
    auditor = LinguisticBiasAuditor()
    
    sample_questions = {
        'English': ["What are essential nutrients?", "When should a woman seek care?"],
        'Luganda': ["Byetaago bya maanyi ki?", "Omukyala ayinza okunoonya obujanjabi ddi?"],
        'Runyankore': ["Ebiri kukozesa nka ki?", "Omukazi ata hairwe okunoonya obujanjabi?"]
    }
    
    inter_df = auditor.analyze_interrogative_structure(sample_questions)
    
    print("\n Linguistic bias audit test complete!")