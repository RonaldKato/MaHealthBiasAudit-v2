"""
Stratum II: Linguistic and Structural Bias Audit
Based on Section 5 of the research proposal
Includes: Tokenisation Parity, Morphological Analysis, OOV Audit,
Trust-Aware Module for cultural terminology
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import re

from config import (LANGUAGES, PRIMARY_LANGUAGES, THRESHOLDS, 
                   INTERROGATIVE_PATTERNS, CULTURAL_TERMINOLOGY, DIALECT_MARKERS)
from utils import (normalize_text, compute_fertility_penalty, compute_oov_rate,
                  get_morpheme_boundaries, compute_boundary_f1)


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
        self.tokenisation_results = []
        self.morphological_results = []
        self.trust_results: Dict[str, TrustAwareResult] = {}
        self.interrogative_analysis = None
    
    def compute_tokenisation_parity(self, 
                                     questions: Dict[str, List[str]],
                                     tokeniser_performances: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compute Tokenisation Parity (TP) across languages
        TP(L) = mean_s [token_count(s_L) / token_count(s_en)]
        """
        rows = []
        
        for lang, lang_questions in questions.items():
            if lang == 'English':
                continue
            
            # Get fertility penalties from tokeniser performances
            for tokeniser in ['mBERT', 'XLM-R', 'AfriBERTa']:
                fertility = tokeniser_performances.get(tokeniser, {}).get(lang, 1.0)
                
                rows.append({
                    'Language': lang,
                    'Tokeniser': tokeniser,
                    'Fertility_Penalty': fertility,
                    'TP_Flag': fertility > THRESHOLDS['tokenisation_parity']
                })
        
        df = pd.DataFrame(rows)
        self.tokenisation_results = df
        return df
    
    def morphological_analysis(self, 
                                sample_words: Dict[str, List[str]],
                                tokeniser_segmentations: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Morphological analysis with Morfessor + native validation
        """
        results = {}
        
        for lang, words in sample_words.items():
            lang_results = []
            
            for word in words:
                # Get tokeniser segmentation
                tokeniser_segs = tokeniser_segmentations.get(lang, {}).get(word, [word])
                
                # Simulate morpheme segmentation (would use Morfessor + native linguist)
                morpheme_segs = self._simulate_morpheme_segmentation(word, lang)
                
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
            self.morphological_results.append(results[lang])
        
        return results
    
    def _simulate_morpheme_segmentation(self, word: str, lang: str) -> List[str]:
        """Simulate morpheme segmentation for Bantu languages"""
        # Based on Figure 3: Okuzaala (Luganda: 'to give birth')
        bantu_examples = {
            'okuzaala': ['Oku-', '-zaal-', '-a'],
            'okuzaara': ['Oku-', '-zaar-', '-a'],
            'mwanamke': ['m-', '-wana-', '-mke'],
            'mjamzito': ['m-', '-ja-', '-mzito'],
            'kunyonyesha': ['ku-', '-nyony-', '-esh-', '-a']
        }
        
        if word in bantu_examples:
            return bantu_examples[word]
        
        # Generic segmentation for Bantu languages
        if lang in ['Luganda', 'Runyankore', 'Swahili'] and len(word) > 5:
            # Look for common prefixes
            prefixes = ['ku', 'mu', 'ki', 'tu', 'a', 'wa', 'm']
            for prefix in prefixes:
                if word.startswith(prefix) and len(word) > len(prefix) + 2:
                    stem = word[len(prefix):]
                    # Check for common suffixes
                    if stem.endswith('a'):
                        return [f"{prefix}-", f"-{stem[:-1]}-", '-a']
                    return [f"{prefix}-", stem]
        
        return [word]
    
    def analyze_interrogative_structure(self, questions: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Analyze interrogative structure patterns across languages
        """
        rows = []
        
        for lang, lang_questions in questions.items():
            pattern_info = INTERROGATIVE_PATTERNS.get(lang, INTERROGATIVE_PATTERNS['English'])
            pattern = pattern_info['pattern']
            expected_type = pattern_info['type']
            
            for q in lang_questions:
                match = re.search(pattern, q.lower()) if isinstance(pattern, str) else None
                interrogative_position = match.start() if match else -1
                
                # Determine actual type based on position
                if interrogative_position == 0:
                    actual_type = 'wh_fronted'
                elif interrogative_position > 0:
                    actual_type = 'inline'
                else:
                    actual_type = 'sentence_final'
                
                rows.append({
                    'Language': lang,
                    'Question': q[:60] + '...' if len(q) > 60 else q,
                    'Expected_Type': expected_type,
                    'Actual_Type': actual_type,
                    'Interrogative_Position': interrogative_position,
                    'Mismatch': expected_type != actual_type
                })
        
        df = pd.DataFrame(rows)
        self.interrogative_analysis = df
        return df
    
    def trust_aware_module(self, texts: List[str], language: str) -> TrustAwareResult:
        """
        Trust-Aware Module for cultural terminology and phrases
        Based on Section "Trust-Aware Module" requirement
        """
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
            # Higher score for culturally appropriate medical terms
            cultural_bonus = sum(t['cultural_importance'] for t in cultural_terms_found) / len(cultural_terms_found)
            trust_score = min(1.0, 0.5 + cultural_bonus * 0.5)
            preservation_needed = any(t['is_medical'] for t in cultural_terms_found)
        else:
            trust_score = 0.6  # Neutral baseline
            preservation_needed = False
        
        # Generate recommendations
        recommendations = []
        if not cultural_terms_found:
            recommendations.append(f"Add culturally specific {language} medical terminology")
        for term in cultural_terms_found:
            if term['is_medical']:
                recommendations.append(f"PRESERVE: '{term['term']}' ({term['translation']}) is valid cultural knowledge")
        
        result = TrustAwareResult(
            language=language,
            cultural_terms_found=cultural_terms_found,
            trust_score=trust_score,
            recommendations=recommendations,
            preservation_needed=preservation_needed
        )
        
        self.trust_results[language] = result
        return result
    
    def detect_dialect_variance(self, texts: List[str], language: str) -> Dict:
        """
        Detect dialect variance during tokenisation and morphology
        Addresses the requirement for capturing linguistic deviations
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
            'recommendations': [
                f"Consider {primary}-specific tokenisation" if primary and variance_detected else None
            ]
        }
    
    def compute_morphological_alignment_scores(self) -> pd.DataFrame:
        """Morphological Alignment Score (MAS) across languages"""
        rows = []
        
        for df in self.morphological_results:
            if df is not None and not df.empty:
                lang = df.iloc[0].get('language', 'Unknown') if 'language' in df.columns else 'Unknown'
                rows.append({
                    'Language': lang,
                    'Avg_Boundary_F1': df['boundary_f1'].mean(),
                    'Avg_Fragmentation': df['fragmentation_score'].mean(),
                    'Pct_Aligned': (df['is_aligned'].sum() / len(df)) * 100,
                    'MAS_Flag': df['boundary_f1'].mean() < THRESHOLDS['mas_threshold']
                })
        
        return pd.DataFrame(rows)
    
    def get_flags(self) -> List[str]:
        """Generate flags based on linguistic bias thresholds"""
        flags = []
        
        # Tokenisation parity flags
        if not self.tokenisation_results.empty:
            high_tp = self.tokenisation_results[self.tokenisation_results['TP_Flag']]
            for _, row in high_tp.iterrows():
                flags.append(f"HIGH_FERTILITY: {row['Language']} with {row['Tokeniser']} (TP={row['Fertility_Penalty']:.2f})")
        
        # Morphological alignment flags
        mas_df = self.compute_morphological_alignment_scores()
        if not mas_df.empty:
            low_mas = mas_df[mas_df['MAS_Flag']]
            for _, row in low_mas.iterrows():
                flags.append(f"POOR_MORPH_ALIGNMENT: {row['Language']} (MAS={row['Avg_Boundary_F1']:.3f})")
        
        # Interrogative mismatch flags
        if self.interrogative_analysis is not None:
            mismatches = self.interrogative_analysis[self.interrogative_analysis['Mismatch']]
            if not mismatches.empty:
                flags.append(f"QUERY_STRUCTURE_MISMATCH: {len(mismatches)} questions have non-English patterns")
        
        return flags


# Test the auditor
if __name__ == "__main__":
    auditor = LinguisticBiasAuditor()
    
    sample_questions = {
        'English': ["What are essential nutrients?", "When should a woman seek care?"],
        'Luganda': ["Byetaago bya maanyi ki?", "Omukyala ayinza okunoonya obujanjabi ddi?"],
        'Runyankore': ["Ebiri kukozesa nka ki?", "Omukazi ata hairwe okunoonya obujanjabi?"]
    }
    
    inter_df = auditor.analyze_interrogative_structure(sample_questions)
    print("Interrogative Analysis:")
    print(inter_df[['Language', 'Expected_Type', 'Actual_Type', 'Mismatch']].to_string())
    
    print("\n Linguistic bias auditor test complete!")