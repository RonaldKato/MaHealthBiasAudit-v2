"""
Stratum III: Model-Level Bias Audit
Includes: SLM fine-tuning, performance metrics, cross-lingual transfer analysis
Based on Section 6 of the research proposal
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIGS, FINE_TUNE_CONDITIONS, THRESHOLDS
from utils import compute_chrf, set_seed, RANDOM_SEED


@dataclass
class ModelPerformance:
    """Performance metrics for a model on a specific language"""
    model_name: str
    language: str
    training_condition: str
    exact_match: float
    token_f1: float
    precision: float
    recall: float
    chr_f2: float
    perplexity: float


@dataclass
class BiasMetrics:
    """Bias metrics derived from model performance"""
    language: str
    f1_disparity: float  # Difference from English baseline
    relative_disparity: float  # Normalized disparity
    transfer_gain: float  # Improvement from language-specific training
    needs_intervention: bool


class ModelBiasAuditor:
    """
    Stratum III: Model-level bias audit and evaluation
    Simulates fine-tuning and evaluates cross-lingual transfer
    """
    
    def __init__(self, device: str = 'cpu'):
        """Initialize model bias auditor"""
        set_seed(RANDOM_SEED)
        self.device = device
        self.performance_records: List[ModelPerformance] = []
        self.bias_metrics: List[BiasMetrics] = []
        
        print(f"✅ Model Bias Auditor initialized (device: {device})")
    
    def simulate_fine_tuning(self, 
                              questions: Dict[str, List[str]],
                              answers: Dict[str, List[str]],
                              model_name: str,
                              training_condition: str,
                              train_langs: List[str]) -> Dict:
        """
        Simulate fine-tuning of a model on specific languages
        In production, this would use actual transformers training
        
        Args:
            questions: Dictionary of questions per language
            answers: Dictionary of answers per language
            model_name: Name of the model (mBERT, XLM-R, AfriBERTa)
            training_condition: FT-EN, FT-SW, FT-YO, FT-AM, FT-MULTI
            train_langs: List of languages to train on
        
        Returns:
            Dictionary with trained model performance estimates
        """
        # Get model configuration
        model_config = MODEL_CONFIGS.get(model_name, {})
        base_performance = model_config.get('fertility_baseline', 1.0)
        
        # Simulate performance based on training condition
        performance = {}
        
        for lang in questions.keys():
            # Base accuracy depends on language resource level
            from config import LANGUAGES
            lang_info = LANGUAGES.get(lang, {})
            resource_level = lang_info.get('resource_level', 'medium')
            
            resource_multiplier = {
                'high': 0.9,
                'medium': 1.0,
                'low': 1.2,
                'very_low': 1.4
            }.get(resource_level, 1.0)
            
            # Morphological complexity penalty
            morph_complexity = lang_info.get('morphological_complexity', 1.0)
            morph_penalty = 1.0 + (morph_complexity - 1.0) * 0.3
            
            # Training condition multiplier
            if training_condition == 'FT-EN':
                # English-only training - worst for other languages
                lang_multiplier = 0.6 if lang != 'English' else 1.0
            elif training_condition == f'FT-{lang[:2].upper()}':
                # Language-specific training - best for that language
                lang_multiplier = 1.0 if lang in train_langs else 0.7
            elif training_condition == 'FT-MULTI':
                # Multilingual training - good for all
                lang_multiplier = 0.85
            else:
                lang_multiplier = 0.7
            
            # Exact match score (0-1)
            exact_match = 0.85 * lang_multiplier / (resource_multiplier * morph_penalty)
            exact_match = max(0.3, min(0.95, exact_match))
            
            # Token F1 score
            token_f1 = exact_match * 0.9
            
            # Precision and recall
            precision = exact_match * 0.85
            recall = exact_match * 0.95
            
            # chrF2 for morphologically rich languages
            if morph_complexity > 1.5:
                chr_f2 = exact_match * 0.85
            else:
                chr_f2 = exact_match * 0.95
            
            performance[lang] = {
                'exact_match': exact_match,
                'token_f1': token_f1,
                'precision': precision,
                'recall': recall,
                'chr_f2': chr_f2,
                'perplexity': 1.0 / max(exact_match, 0.1)
            }
        
        return performance
    
    def evaluate_model(self, 
                       model_name: str,
                       training_condition: str,
                       train_langs: List[str],
                       questions: Dict[str, List[str]],
                       answers: Dict[str, List[str]]) -> List[ModelPerformance]:
        """
        Evaluate a model on all languages
        """
        # Simulate fine-tuning
        performance = self.simulate_fine_tuning(questions, answers, model_name, 
                                                  training_condition, train_langs)
        
        # Create performance records
        records = []
        for lang, perf in performance.items():
            records.append(ModelPerformance(
                model_name=model_name,
                language=lang,
                training_condition=training_condition,
                exact_match=perf['exact_match'],
                token_f1=perf['token_f1'],
                precision=perf['precision'],
                recall=perf['recall'],
                chr_f2=perf['chr_f2'],
                perplexity=perf['perplexity']
            ))
        
        self.performance_records.extend(records)
        return records
    
    def run_experiment_matrix(self,
                               questions: Dict[str, List[str]],
                               answers: Dict[str, List[str]],
                               models: List[str] = None) -> pd.DataFrame:
        """
        Run the full 5x3 experiment matrix
        5 training conditions × 3 models = 15 experiments
        
        Args:
            questions: Dictionary of questions per language
            answers: Dictionary of answers per language
            models: List of model names (default: ['mBERT', 'XLM-R', 'AfriBERTa'])
        
        Returns:
            DataFrame with all performance results
        """
        print("\n" + "="*70)
        print("🤖 STRATUM III: Model-Level Bias Audit")
        print("="*70)
        print("\n📊 Running 5x3 Experiment Matrix (15 experiments)")
        print("-" * 50)
        
        if models is None:
            models = ['mBERT', 'XLM-R', 'AfriBERTa']
        
        # Define training conditions
        training_conditions = [
            ('FT-EN', ['English']),
            ('FT-SW', ['Swahili']),
            ('FT-YO', ['Yoruba']),
            ('FT-AM', ['Amharic']),
            ('FT-MULTI', list(questions.keys()))
        ]
        
        all_results = []
        
        for model_name in models:
            print(f"\n📱 Model: {model_name}")
            print("-" * 30)
            
            for condition, train_langs in training_conditions:
                results = self.evaluate_model(model_name, condition, train_langs,
                                               questions, answers)
                all_results.extend(results)
                
                # Print quick summary
                avg_f1 = np.mean([r.token_f1 for r in results])
                print(f"   {condition}: avg F1={avg_f1:.3f}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame([vars(r) for r in all_results])
        
        print(f"\n✅ Experiment complete! {len(results_df)} performance records")
        
        return results_df
    
    def compute_bias_metrics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute bias metrics from performance results:
        - F1 disparity (difference from English baseline)
        - Transfer gain (improvement from language-specific training)
        """
        print("\n" + "="*60)
        print("📊 Computing Model Bias Metrics")
        print("="*60)
        
        bias_metrics = []
        
        # Find English baseline performance for each model
        eng_baseline = {}
        for model in results_df['model_name'].unique():
            eng_data = results_df[(results_df['model_name'] == model) & 
                                  (results_df['language'] == 'English')]
            if not eng_data.empty:
                eng_baseline[model] = eng_data['token_f1'].iloc[0]
            else:
                eng_baseline[model] = 0.85  # Default
        
        # Compute metrics for each language-model combination
        for model in results_df['model_name'].unique():
            baseline_f1 = eng_baseline[model]
            
            for lang in results_df['language'].unique():
                if lang == 'English':
                    continue
                
                # Find English-trained performance
                eng_trained = results_df[(results_df['model_name'] == model) &
                                         (results_df['language'] == lang) &
                                         (results_df['training_condition'] == 'FT-EN')]
                
                # Find language-specific trained performance
                lang_trained = results_df[(results_df['model_name'] == model) &
                                          (results_df['language'] == lang) &
                                          (results_df['training_condition'] == f'FT-{lang[:2].upper()}')]
                
                if not eng_trained.empty and not lang_trained.empty:
                    eng_f1 = eng_trained['token_f1'].iloc[0]
                    lang_f1 = lang_trained['token_f1'].iloc[0]
                    
                    f1_disparity = baseline_f1 - eng_f1
                    relative_disparity = f1_disparity / max(baseline_f1, 0.01)
                    transfer_gain = lang_f1 - eng_f1
                    
                    needs_intervention = (f1_disparity > THRESHOLDS['f1_disparity_high'] or 
                                         transfer_gain > 0.15)
                    
                    bias_metrics.append(BiasMetrics(
                        language=lang,
                        f1_disparity=f1_disparity,
                        relative_disparity=relative_disparity,
                        transfer_gain=transfer_gain,
                        needs_intervention=needs_intervention
                    ))
        
        self.bias_metrics = bias_metrics
        
        # Print summary
        print(f"\n   Bias Metrics Summary:")
        for metric in bias_metrics:
            status = "⚠️ NEEDS INTERVENTION" if metric.needs_intervention else "✓ OK"
            print(f"      {metric.language}: F1 disparity={metric.f1_disparity:.3f}, "
                  f"transfer gain={metric.transfer_gain:.3f} [{status}]")
        
        return pd.DataFrame([vars(m) for m in bias_metrics])
    
    def identify_cross_lingual_transfer_issues(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify cross-lingual transfer issues
        Measures how well knowledge transfers from English to other languages
        """
        print("\n" + "="*60)
        print("🔄 Analyzing Cross-Lingual Transfer")
        print("="*60)
        
        transfer_issues = []
        
        for model in results_df['model_name'].unique():
            for lang in results_df['language'].unique():
                if lang == 'English':
                    continue
                
                # Get English-trained performance
                eng_trained = results_df[(results_df['model_name'] == model) &
                                         (results_df['language'] == lang) &
                                         (results_df['training_condition'] == 'FT-EN')]
                
                # Get multilingual-trained performance
                multi_trained = results_df[(results_df['model_name'] == model) &
                                           (results_df['language'] == lang) &
                                           (results_df['training_condition'] == 'FT-MULTI')]
                
                if not eng_trained.empty and not multi_trained.empty:
                    eng_f1 = eng_trained['token_f1'].iloc[0]
                    multi_f1 = multi_trained['token_f1'].iloc[0]
                    
                    transfer_efficiency = multi_f1 / max(eng_f1, 0.01)
                    improvement = multi_f1 - eng_f1
                    
                    transfer_issues.append({
                        'Model': model,
                        'Language': lang,
                        'English_Trained_F1': round(eng_f1, 4),
                        'Multilingual_Trained_F1': round(multi_f1, 4),
                        'Transfer_Efficiency': round(transfer_efficiency, 3),
                        'Improvement': round(improvement, 4),
                        'Issue_Severity': 'High' if transfer_efficiency < 0.7 else 
                                         'Moderate' if transfer_efficiency < 0.85 else 'Low'
                    })
        
        df = pd.DataFrame(transfer_issues)
        
        print(f"\n   Transfer Analysis Summary:")
        for _, row in df.iterrows():
            severity_icon = '🔴' if row['Issue_Severity'] == 'High' else '🟡' if row['Issue_Severity'] == 'Moderate' else '🟢'
            print(f"      {severity_icon} {row['Model']} → {row['Language']}: "
                  f"efficiency={row['Transfer_Efficiency']:.2f}")
        
        return df
    
    def get_flags(self) -> List[str]:
        """Generate flags based on model bias metrics"""
        flags = []
        
        for metric in self.bias_metrics:
            if metric.needs_intervention:
                if metric.f1_disparity > THRESHOLDS['f1_disparity_high']:
                    flags.append(f"HIGH_F1_DISPARITY: {metric.language} = {metric.f1_disparity:.3f}")
                if metric.transfer_gain > 0.15:
                    flags.append(f"HIGH_TRANSFER_GAIN: {metric.language} could improve with language-specific training (+{metric.transfer_gain:.1%})")
        
        return flags
    
    def run_full_audit(self, questions: Dict[str, List[str]], 
                        answers: Dict[str, List[str]]) -> Dict:
        """
        Run complete model bias audit
        
        Args:
            questions: Dictionary of questions per language
            answers: Dictionary of answers per language
        
        Returns:
            Dictionary with all audit results
        """
        print("\n" + "="*70)
        print("🤖 STRATUM III: Model Bias Audit")
        print("="*70)
        
        # Run experiment matrix
        results_df = self.run_experiment_matrix(questions, answers)
        
        # Compute bias metrics
        bias_metrics_df = self.compute_bias_metrics(results_df)
        
        # Identify cross-lingual transfer issues
        transfer_issues_df = self.identify_cross_lingual_transfer_issues(results_df)
        
        # Generate flags
        flags = self.get_flags()
        
        return {
            'performance_results': results_df,
            'bias_metrics': bias_metrics_df,
            'transfer_issues': transfer_issues_df,
            'flags': flags
        }


# Test the auditor
if __name__ == "__main__":
    auditor = ModelBiasAuditor()
    
    sample_questions = {
        'English': ["What are essential nutrients?", "What are signs of labor?"],
        'Swahili': ["Virutubisho muhimu?", "Dalili za uchungu?"],
        'Yoruba': ["Awọn eroja pataki?", "Awọn ami ibi?"]
    }
    
    sample_answers = {
        'English': ["Folic acid, iron.", "Contractions, water breaking."],
        'Swahili': ["Asidi foliki, chuma.", "Mikazo, kupasuka kwa maji."],
        'Yoruba': ["Folic acid, iron.", "Ìrora, omi ìyá."]
    }
    
    results = auditor.run_full_audit(sample_questions, sample_answers)
    
    print("\n✅ Model bias audit test complete!")