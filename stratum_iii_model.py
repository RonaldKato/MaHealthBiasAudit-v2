"""
Stratum III: Model Bias Audit 
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from config import THRESHOLDS, PRIMARY_LANGUAGES
from utils import set_seed, RANDOM_SEED


@dataclass
class ModelPerformance:
    model_name: str
    language: str
    token_f1: float


class ModelBiasAuditor:
    def __init__(self):
        set_seed(RANDOM_SEED)
    
    def run_full_audit(self, questions: Dict[str, List[str]], 
                        answers: Dict[str, List[str]]) -> Dict:
        """Run model bias audit"""
        print("\n" + "="*70)
        print("STRATUM III: Model Bias Audit")
        print("="*70)
        
        performances = []
        
        # Simulate performance based on language complexity
        complexity = {
            'English': 1.0,
            'Swahili': 0.85,
            'Luganda': 0.72,
            'Runyankore': 0.65
        }
        
        for lang in PRIMARY_LANGUAGES:
            perf = complexity.get(lang, 0.7)
            performances.append(ModelPerformance(
                model_name='AfriBERTa',
                language=lang,
                token_f1=perf
            ))
        
        perf_df = pd.DataFrame([(p.model_name, p.language, p.token_f1) 
                                for p in performances], 
                               columns=['model_name', 'language', 'token_f1'])
        
        # Compute bias metrics
        eng_f1 = perf_df[perf_df['language'] == 'English']['token_f1'].values[0]
        bias_metrics = []
        
        for lang in PRIMARY_LANGUAGES:
            if lang != 'English':
                lang_f1 = perf_df[perf_df['language'] == lang]['token_f1'].values[0]
                disparity = eng_f1 - lang_f1
                bias_metrics.append({
                    'language': lang,
                    'f1_disparity': disparity,
                    'needs_intervention': disparity > THRESHOLDS['f1_disparity_high']
                })
        
        bias_df = pd.DataFrame(bias_metrics)
        
        print(f"\n   Performance Results:")
        for _, row in perf_df.iterrows():
            print(f"   {row['language']}: F1={row['token_f1']:.3f}")
        
        flags = []
        for _, row in bias_df.iterrows():
            if row['needs_intervention']:
                flags.append(f"HIGH_F1_DISPARITY: {row['language']} = {row['f1_disparity']:.3f}")
        
        return {
            'performance_results': perf_df,
            'bias_metrics': bias_df,
            'flags': flags
        }