"""
Cross-Lingual Evaluation Engine
SDI, RCA Cascade, Bias Patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

from config import THRESHOLDS, PRIMARY_LANGUAGES, LANGUAGES, RCA_COLORS
from utils import set_seed, RANDOM_SEED

@dataclass
class RCAResult:
    language_pair: Tuple[str, str]
    question_idx: int
    sdi_value: float
    root_cause: str
    preserve: bool


class CrossLingualEvaluator:
    """Cross-lingual evaluation with RCA cascade"""
    
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.sdi_matrix = None
        self.rca_results = []
    
    def compute_semantic_divergence_index(self, 
                                           embeddings: Dict[str, np.ndarray],
                                           questions: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute SDI for all language pairs"""
        print("\n" + "="*70)
        print("Semantic Divergence Index (SDI)")
        print("="*70)
        
        languages = PRIMARY_LANGUAGES
        n = len(languages)
        sdi_matrix = pd.DataFrame(np.zeros((n, n)), index=languages, columns=languages)
        
        pair_sdis = []
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i == j:
                    sdi_matrix.loc[lang1, lang2] = 0.0
                    continue
                
                emb1 = embeddings.get(lang1, np.array([]))
                emb2 = embeddings.get(lang2, np.array([]))
                
                if len(emb1) == 0 or len(emb2) == 0:
                    sdi_matrix.loc[lang1, lang2] = 0.5
                    continue
                
                min_len = min(len(emb1), len(emb2))
                sims = []
                for k in range(min_len):
                    sim = cosine_similarity(emb1[k:k+1], emb2[k:k+1])[0][0]
                    sims.append(sim)
                
                mean_sim = np.mean(sims) if sims else 0
                sdi = 1 - mean_sim
                sdi_matrix.loc[lang1, lang2] = sdi
                sdi_matrix.loc[lang2, lang1] = sdi
                
                pair_sdis.append({
                    'pair': f"{lang1}-{lang2}",
                    'lang1': lang1,
                    'lang2': lang2,
                    'sdi': sdi
                })
        
        self.sdi_matrix = sdi_matrix
        
        # Print rankings
        sorted_pairs = sorted(pair_sdis, key=lambda x: x['sdi'], reverse=True)
        print("\nSDI RANKINGS (Highest to Lowest Bias):")
        for rank, item in enumerate(sorted_pairs, 1):
            severity = "HIGH" if item['sdi'] > THRESHOLDS['sdi_high'] else \
                      "MODERATE" if item['sdi'] > THRESHOLDS['sdi_moderate'] else "🟢 LOW"
            print(f"   {rank}. {item['pair']}: {item['sdi']:.3f} [{severity}]")
        
        # Language-specific bias
        print("\n LANGUAGE BIAS (vs English):")
        for lang in languages:
            if lang != 'English':
                sdi = sdi_matrix.loc['English', lang]
                icon = "⚠️" if sdi > 0.35 else "✓"
                print(f"   {icon} {lang}: SDI={sdi:.3f}")
        
        return sdi_matrix
    
    def root_cause_attribution_cascade(self,
                                         sdi_matrix: pd.DataFrame,
                                         questions: Dict[str, List[str]],
                                         answers: Dict[str, List[str]],
                                         dataset_name: str = "Private") -> List[RCAResult]:
        """RCA Cascade: Tokenisation → Query Structure → Cultural → Morphology"""
        print("\n" + "="*70)
        print(f"RCA Cascade - {dataset_name}")
        print("="*70)
        
        self.rca_results = []
        cause_counts = {'TOKENISATION': 0, 'QUERY_STRUCTURE': 0, 'CULTURAL': 0, 'MORPHOLOGY': 0}
        
        for i, lang1 in enumerate(PRIMARY_LANGUAGES):
            for j, lang2 in enumerate(PRIMARY_LANGUAGES):
                if i >= j:
                    continue
                
                sdi = sdi_matrix.loc[lang1, lang2]
                
                # Determine root cause based on language characteristics
                morph1 = LANGUAGES[lang1]['morphological_complexity']
                morph2 = LANGUAGES[lang2]['morphological_complexity']
                
                if abs(morph1 - morph2) > 1.0:
                    cause = 'TOKENISATION'
                elif lang1 in ['Luganda', 'Runyankore'] or lang2 in ['Luganda', 'Runyankore']:
                    cause = 'QUERY_STRUCTURE'
                elif lang1 in ['Luganda', 'Runyankore', 'Swahili'] and lang2 == 'English':
                    cause = 'CULTURAL'
                else:
                    cause = 'MORPHOLOGY'
                
                preserve = cause == 'CULTURAL'
                cause_counts[cause] += 1
                
                self.rca_results.append(RCAResult(
                    language_pair=(lang1, lang2),
                    question_idx=0,
                    sdi_value=sdi,
                    root_cause=cause,
                    preserve=preserve
                ))
        
        # Print distribution
        total = sum(cause_counts.values())
        print(f"\n RCA Distribution:")
        for cause, count in cause_counts.items():
            pct = count / total * 100
            print(f"   {cause}: {pct:.1f}%")
        
        return self.rca_results
    
    def compute_mrr(self, embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Compute Mean Reciprocal Rank for cross-lingual retrieval"""
        print("\n" + "="*60)
        print("Mean Reciprocal Rank (MRR)")
        print("="*60)
        
        languages = PRIMARY_LANGUAGES
        mrr_matrix = pd.DataFrame(np.zeros((len(languages), len(languages))),
                                  index=languages, columns=languages)
        
        for src in languages:
            for tgt in languages:
                if src == tgt:
                    mrr_matrix.loc[src, tgt] = 1.0
                    continue
                
                src_emb = embeddings.get(src, np.array([]))
                tgt_emb = embeddings.get(tgt, np.array([]))
                
                if len(src_emb) == 0 or len(tgt_emb) == 0:
                    mrr_matrix.loc[src, tgt] = 0.5
                    continue
                
                min_len = min(len(src_emb), len(tgt_emb))
                ranks = []
                
                for k in range(min_len):
                    query = src_emb[k:k+1]
                    sims = cosine_similarity(query, tgt_emb)[0]
                    correct_sim = sims[k]
                    rank = np.sum(sims > correct_sim) + 1
                    ranks.append(1.0 / rank)
                
                mrr = np.mean(ranks) if ranks else 0
                mrr_matrix.loc[src, tgt] = mrr
        
        print(f"\n Average MRR: {mrr_matrix.values[np.triu_indices_from(mrr_matrix.values, k=1)].mean():.3f}")
        
        return mrr_matrix
    
    def get_flags(self) -> List[str]:
        """Generate flags from cross-lingual evaluation"""
        flags = []
        
        if self.sdi_matrix is not None:
            for lang in PRIMARY_LANGUAGES:
                if lang != 'English':
                    sdi = self.sdi_matrix.loc['English', lang]
                    if sdi > THRESHOLDS['sdi_high']:
                        flags.append(f"HIGH_SDI: English-{lang} = {sdi:.3f}")
        
        for result in self.rca_results:
            if result.preserve:
                flags.append(f"PRESERVE_CULTURAL: {result.language_pair[0]}-{result.language_pair[1]}")
        
        return flags
    
    def run_full_evaluation(self, embeddings: Dict[str, np.ndarray],
                            questions: Dict[str, List[str]],
                            answers: Dict[str, List[str]]) -> Dict:
        """Run complete cross-lingual evaluation"""
        print("\n" + "="*70)
        print("STRATUM III: Cross-Lingual Evaluation")
        print("="*70)
        
        sdi = self.compute_semantic_divergence_index(embeddings, questions)
        rca = self.root_cause_attribution_cascade(sdi, questions, answers)
        mrr = self.compute_mrr(embeddings)
        
        return {
            'sdi_matrix': sdi,
            'rca_results': rca,
            'mrr_matrix': mrr,
            'flags': self.get_flags()
        }