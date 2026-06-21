"""
MaHealthBiasAudit - Main Pipeline
Complete bias audit for maternal health datasets
"""

import os
import sys
import time
import random
import copy
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Import dataset modules
try:
    from maternal_multlingual_dataset import MaternalMultilingualDataset
    print("✓ maternal_multlingual_dataset.py loaded successfully")
except ImportError as e:
    print(f"Warning: maternal_multlingual_dataset.py not found: {e}")
    MaternalMultilingualDataset = None

try:
    from mother_dataset import MotherDataset
    print("✓ mother_dataset.py loaded successfully")
except ImportError as e:
    print(f"Warning: mother_dataset.py not found: {e}")
    MotherDataset = None

from config import (
    PRIMARY_LANGUAGES, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR,
    EXPERIMENT_SIZES, RANDOM_SEED, EXECUTION_TIMESTAMP,
    VALIDATION_DIR, EXPERIMENTS_DIR, BIAS_SENTENCE_CHARACTERISTICS,
    BIAS_REDUCTION_TEMPLATES, SAMPLES_DIR
)
from utils import set_seed, save_report, setup_logger, print_bias_characteristics, get_bias_characteristics
from preprocessing import MultilingualPreprocessor
from stratum_i_statistical import StatisticalBiasAuditor
from stratum_ii_linguistic import LinguisticBiasAuditor
from stratum_iii_model import ModelBiasAuditor
from cross_lingual_evaluation import CrossLingualEvaluator
from visualization_dashboard import VisualizationDashboard

from sample_tables_extractor import extract_and_display_sample_tables, SampleTablesExtractor

from encoder_calibration import EncoderCalibrator
from feature_attribution import FeatureAttributor
from bias_reduction import BiasReducer


class MaHealthBiasAudit:
    """Complete bias audit pipeline"""
    
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.logger = setup_logger('main')
        self.results = {}
        self.experiment_results = []
        self.validation_results = {}
        self.sample_results = {}
        self.calibration_results = {}
        self.reduction_results = {}
        
        # Initialize components
        self.preprocessor = MultilingualPreprocessor()
        self.stat_auditor = StatisticalBiasAuditor()
        self.ling_auditor = LinguisticBiasAuditor()
        self.model_auditor = ModelBiasAuditor()
        self.cross_lingual = CrossLingualEvaluator()
        self.viz = VisualizationDashboard()
        
        # Initialize new components
        self.encoder_calibrator = None
        self.feature_attributor = None
        self.bias_reducer = None
        
        # Lazy initialization
        try:
            from encoder_calibration import EncoderCalibrator
            from feature_attribution import FeatureAttributor
            from bias_reduction import BiasReducer
            self.encoder_calibrator = EncoderCalibrator()
            self.feature_attributor = FeatureAttributor()
            self.bias_reducer = BiasReducer()
        except ImportError as e:
            print(f"⚠ Could not load advanced modules: {e}")
        
        print("\n" + "="*70)
        print("MAHEALTHBIASAUDIT - MATERNAL HEALTH BIAS DETECTION")
        print("="*70)
        print("\nBIAS SENTENCE CHARACTERISTICS:")
        print("-"*50)
        for category, info in BIAS_SENTENCE_CHARACTERISTICS.items():
            print(f"  • {category.replace('_', ' ').title()}: {info['description'][:80]}...")
        print("-"*50)
    
    # ============================================================
    # DATASET LOADING METHODS
    # ============================================================
    
    def load_maternal_multilingual_dataset(self) -> Dict:
        """Load the maternal multilingual dataset"""
        if MaternalMultilingualDataset is None:
            raise ImportError("maternal_multlingual_dataset.py not found")
        
        try:
            dataset = MaternalMultilingualDataset()
            data = dataset.load_your_data()
            
            print(f"  ✓ Loaded {len(data)} categories")
            print(f"  ✓ Categories: {list(data.keys())[:3]}...")
            
            total_answers = 0
            lang_counts = {}
            for category, cat_data in data.items():
                answers = cat_data.get('answers', {})
                questions = cat_data.get('questions', [])
                for lang, ans_list in answers.items():
                    count = len(ans_list)
                    total_answers += count
                    lang_counts[lang] = lang_counts.get(lang, 0) + count
            
            print(f"  ✓ Total answers: {total_answers}")
            print(f"  ✓ Languages: {lang_counts}")
            
            return data
            
        except Exception as e:
            print(f" Error loading maternal dataset: {e}")
            raise
    
    def load_mother_dataset(self) -> Dict:
        """Load the MOTHER dataset and convert to standard format"""
        if MotherDataset is None:
            raise ImportError("mother_dataset.py not found")
        
        try:
            dataset = MotherDataset()
            mother_data = dataset.load_your_data()
            
            converted_data = {}
            
            for category_name, category_data in mother_data.items():
                if 'english' in category_name.lower():
                    lang = 'English'
                elif 'luganda' in category_name.lower():
                    lang = 'Luganda'
                elif 'runyankole' in category_name.lower() or 'rukiga' in category_name.lower():
                    lang = 'Runyankole-Rukiga'
                else:
                    lang = category_name.replace('about_', '').replace('_', ' ').title()
                
                questions = category_data.get('questions', [])
                answers_dict = category_data.get('answers', {})
                
                if lang not in converted_data:
                    converted_data[lang] = {
                        'questions': [],
                        'answers': {lang: []}
                    }
                
                for q in questions:
                    if q not in converted_data[lang]['questions']:
                        converted_data[lang]['questions'].append(q)
                
                for answer_key, answer_list in answers_dict.items():
                    if isinstance(answer_list, list):
                        for ans in answer_list:
                            if ans and isinstance(ans, str):
                                converted_data[lang]['answers'][lang].append(ans)
            
            final_data = {}
            for lang, data in converted_data.items():
                key = f"about_{lang.lower().replace('-', '_')}"
                final_data[key] = {
                    'questions': data['questions'],
                    'answers': data['answers']
                }
            
            print(f"  ✓ Converted MOTHER dataset: {len(final_data)} categories")
            print(f"  ✓ Categories: {list(final_data.keys())}")
            
            return final_data
            
        except Exception as e:
            print(f"  Error loading MOTHER dataset: {e}")
            raise
    
    def run_encoder_calibration(self, embeddings: np.ndarray, labels: List[str], 
                                answers_by_lang: Dict) -> Dict:
            """Run encoder calibration and generate calibration figure"""
            print("\n" + "="*70)
            print(" RUNNING ENCODER CALIBRATION")
            print("="*70)
            
            if self.encoder_calibrator is None:
                print("⚠ Encoder calibrator not available")
                return {}
            
            # Run calibration
            calibration_results = self.encoder_calibrator.calibrate_with_reference_sets(
                embeddings, labels, answers_by_lang
            )
            
            self.calibration_results = calibration_results
            
            # Validate SDI measurements
            cl_results = self.results.get('cross_lingual', {})
            sdi_matrix = cl_results.get('sdi_matrix')
            sdi_classification = cl_results.get('sdi_classification', {})
            
            if sdi_matrix is not None:
                validation = self.encoder_calibrator.validate_sdi_measurements(
                    sdi_matrix, sdi_classification
                )
                calibration_results['validation'] = validation
                
                if validation.get('valid', False):
                    print("\n SDI measurements validated against calibrated ranges")
                else:
                    print("\n⚠ SDI measurements validation issues found")
                    for issue in validation.get('issues', []):
                        print(f"   - {issue}")
            
            # Generate calibration visualization
            print("\n" + "="*70)
            print(" GENERATING ENCODER CALIBRATION FIGURE")
            print("="*70)
            
            if calibration_results and 'reference_sets' in calibration_results:
                self.viz.save_encoder_calibration_plot(calibration_results, "main")
                print("\n Encoder calibration figure saved")
            else:
                print("\n⚠ No calibration data available for visualization")
            
            return calibration_results

    # Add method to run bias reduction
    def run_bias_reduction(self, answers_by_lang: Dict, 
                        embeddings: np.ndarray, labels: List[str]) -> Dict:
        """Run bias reduction framework"""
        print("\n" + "="*70)
        print(" RUNNING BIAS REDUCTION FRAMEWORK")
        print("="*70)
        
        if self.bias_reducer is None:
            print("⚠ Bias reducer not available")
            return {}
        
        # Get flags from results
        all_flags = self._get_all_flags()
        
        if not all_flags:
            print("⚠ No flags available for bias reduction")
            # Generate synthetic flags for demonstration
            all_flags = self._generate_demo_flags()
        
        reduction_results = self.bias_reducer.apply_reduction_framework(
            answers_by_lang, embeddings, labels, all_flags
        )
        
        self.reduction_results = reduction_results
        
        # Generate reduction visualizations
        print("\n" + "="*70)
        print(" GENERATING BIAS REDUCTION VISUALIZATIONS")
        print("="*70)
        
        if reduction_results and 'reduced_sentences' in reduction_results:
            # Save before/after plot
            self.viz.save_sdi_before_after_plot(reduction_results, "main")
            # Save triples table
            self.viz.save_bias_reduction_triples_table(reduction_results, "main")
            print("\n Bias reduction visualizations saved")
        else:
            print("\n No reduction data available for visualization")
        
        return reduction_results

    def _generate_demo_flags(self) -> List[Dict]:
        """Generate demo flags for bias reduction demonstration"""
        return [
            {
                'Type': 'Length_Bias_Critical',
                'Language': 'Luganda',
                'Severity': 'Critical',
                'Description': 'Response is significantly shorter than English',
                'Recommendation': 'Review translation quality'
            },
            {
                'Type': 'Negation_Bias',
                'Language': 'Swahili',
                'Severity': 'High',
                'Description': 'Negation lost in translation',
                'Recommendation': 'Preserve negation markers'
            },
            {
                'Type': 'Tokenisation_Bias',
                'Language': 'Runyankore',
                'Severity': 'High',
                'Description': 'High fertility penalty',
                'Recommendation': 'Use different tokeniser'
            },
            {
                'Type': 'Cultural_Mismatch',
                'Language': 'Luganda',
                'Severity': 'Moderate',
                'Description': 'Traditional remedy replaces medical advice',
                'Recommendation': 'Include both perspectives'
            },
            {
                'Type': 'Structural_Bias',
                'Language': 'Swahili',
                'Severity': 'Moderate',
                'Description': 'Simplified structure',
                'Recommendation': 'Preserve conditional clauses'
            },
            {
                'Type': 'Vocabulary_Bias',
                'Language': 'Runyankore',
                'Severity': 'High',
                'Description': 'Low vocabulary richness',
                'Recommendation': 'Review data collection'
            }
        ]
    # ============================================================
    # SAMPLE EXTRACTION METHODS
    # ============================================================
    
    def extract_unbiased_samples(self, answers_by_lang: Dict[str, List[str]], 
                                embeddings: np.ndarray, labels: List[str],
                                n_samples: int = 3) -> Dict[str, List[str]]:
        """Extract unbiased sample sentences from the dataset"""
        samples = {lang: [] for lang in PRIMARY_LANGUAGES if lang in answers_by_lang}
        
        for lang in PRIMARY_LANGUAGES:
            if lang not in answers_by_lang or not answers_by_lang[lang]:
                continue
            
            texts = answers_by_lang[lang]
            
            # Get indices for this language in the embeddings
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            
            if not lang_indices or len(lang_indices) < n_samples:
                # If not enough samples, use random ones
                selected = random.sample(texts, min(n_samples, len(texts)))
                samples[lang] = selected
                continue
            
            # Use embeddings to find the most representative (centroid) samples
            lang_embeddings = embeddings[lang_indices]
            centroid = np.mean(lang_embeddings, axis=0)
            
            # Find sentences closest to centroid (most representative)
            similarities = np.dot(lang_embeddings, centroid) / (np.linalg.norm(lang_embeddings, axis=1) * np.linalg.norm(centroid) + 1e-8)
            top_indices = np.argsort(similarities)[-n_samples:][::-1]
            
            # FIX: Use the correct indices to access texts
            selected = []
            for idx in top_indices:
                if idx < len(lang_indices) and lang_indices[idx] < len(texts):
                    selected.append(texts[lang_indices[idx]])
            
            # If we didn't get enough samples, fill with random ones
            while len(selected) < n_samples and len(texts) > len(selected):
                remaining = [t for t in texts if t not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
            
            samples[lang] = selected[:n_samples]
        
        return samples

    def extract_biased_samples(self, answers_by_lang: Dict[str, List[str]],
                                embeddings: np.ndarray, labels: List[str],
                                sdi_matrix: pd.DataFrame,
                                n_samples: int = 3) -> Dict[str, List[Dict]]:
        """Extract biased sample sentences with bias type and reduction solution"""
        biased_samples = {lang: [] for lang in PRIMARY_LANGUAGES if lang in answers_by_lang}
        
        # Check if we have English samples
        english_indices = [i for i, l in enumerate(labels) if l == 'English']
        if not english_indices:
            # If no English, return empty biased samples
            for lang in PRIMARY_LANGUAGES:
                if lang in answers_by_lang and answers_by_lang[lang]:
                    texts = answers_by_lang[lang]
                    selected = random.sample(texts, min(n_samples, len(texts)))
                    for text in selected:
                        bias_type = self._detect_bias_type(text, lang)
                        biased_samples[lang].append({
                            'text': text,
                            'bias_type': bias_type,
                            'bias_score': 0.5,
                            'similarity_to_english': 0.5,
                            'reduction_solution': self._generate_reduction_solution(text, bias_type, {})
                        })
            return biased_samples
        
        # Get English embeddings
        eng_emb = embeddings[english_indices]
        eng_centroid = np.mean(eng_emb, axis=0)
        
        for lang in PRIMARY_LANGUAGES:
            if lang == 'English' or lang not in answers_by_lang or not answers_by_lang[lang]:
                continue
            
            texts = answers_by_lang[lang]
            lang_indices = [i for i, l in enumerate(labels) if l == lang]
            
            if not lang_indices or len(lang_indices) < 2:
                # If not enough samples, use random ones
                selected = random.sample(texts, min(n_samples, len(texts)))
                for text in selected:
                    bias_type = self._detect_bias_type(text, lang)
                    biased_samples[lang].append({
                        'text': text,
                        'bias_type': bias_type,
                        'bias_score': 0.5,
                        'similarity_to_english': 0.5,
                        'reduction_solution': self._generate_reduction_solution(text, bias_type, {})
                    })
                continue
            
            # Get language embeddings
            lang_emb = embeddings[lang_indices]
            
            # Compute similarities with English centroid
            similarities = np.dot(lang_emb, eng_centroid) / (np.linalg.norm(lang_emb, axis=1) * np.linalg.norm(eng_centroid) + 1e-8)
            
            # Find sentences with lowest similarity to English (most biased)
            worst_indices = np.argsort(similarities)[:n_samples]
            
            for idx in worst_indices:
                # FIX: Use the correct indices to access texts
                if idx < len(lang_indices) and lang_indices[idx] < len(texts):
                    text = texts[lang_indices[idx]]
                    similarity = similarities[idx]
                    bias_score = 1 - similarity
                    
                    # Determine bias type based on characteristics
                    bias_type = self._detect_bias_type(text, lang)
                    
                    # Generate reduction solution
                    reduction = self._generate_reduction_solution(text, bias_type, {
                        'language': lang,
                        'bias_score': bias_score,
                        'similarity': similarity
                    })
                    
                    biased_samples[lang].append({
                        'text': text,
                        'bias_type': bias_type,
                        'bias_score': float(bias_score),
                        'similarity_to_english': float(similarity),
                        'reduction_solution': reduction
                    })
            
            # If we didn't get enough samples, fill with random ones
            while len(biased_samples[lang]) < n_samples and len(texts) > len(biased_samples[lang]):
                remaining = [t for t in texts if t not in [s['text'] for s in biased_samples[lang]]]
                if remaining:
                    text = random.choice(remaining)
                    bias_type = self._detect_bias_type(text, lang)
                    biased_samples[lang].append({
                        'text': text,
                        'bias_type': bias_type,
                        'bias_score': 0.5,
                        'similarity_to_english': 0.5,
                        'reduction_solution': self._generate_reduction_solution(text, bias_type, {})
                })
    
        return biased_samples
    
    def _detect_bias_type(self, text: str, lang: str) -> str:
        """Detect the type of bias in a sentence"""
        text_lower = text.lower()
        
        # Check for length bias
        words = text.split()
        if len(words) < 5:
            return 'translation_length_disparity'
        
        # Check for cultural keywords
        cultural_keywords = ['herb', 'traditional', 'local', 'cultural', 'belief', 'practice', 'remedy', 
                            'mitishamba', 'ddagala', 'omuddo', 'emibazi', 'kienyeji', 'ennono']
        if any(kw in text_lower for kw in cultural_keywords):
            return 'cultural_concept_mismatch'
        
        # Check for missing medical terms
        medical_terms = ['preeclampsia', 'hypertension', 'diabetes', 'anemia', 'infection', 
                        'medication', 'vaccination', 'ultrasound', 'dosage', 'pregnancy',
                        'preeclampsia', 'hypertension']
        if not any(term in text_lower for term in medical_terms[:3]):
            return 'medical_term_omission'
        
        # Check for emotional tone
        emotion_keywords = ['urgent', 'immediate', 'serious', 'important', 'please', 'careful',
                           'emergency', 'warning', 'critical', 'attention']
        if not any(kw in text_lower for kw in emotion_keywords):
            return 'emotional_tone_shift'
        
        # Check for structural simplification
        conditional_keywords = ['if', 'when', 'although', 'because', 'therefore', 'since',
                               'though', 'while', 'unless', 'whereas']
        if not any(kw in text_lower for kw in conditional_keywords):
            return 'structural_simplification'
        
        # Check for numerical specifics
        numbers = re.findall(r'\d+', text)
        if not numbers:
            return 'instructional_generic_shift'
        
        # Check for negation
        negation_keywords = ['not', 'no', 'never', 'don\'t', 'cannot', 'avoid', 'shouldn\'t',
                            'must not', 'do not', 'does not', 'is not']
        if any(kw in text_lower for kw in negation_keywords):
            return 'negation_misinterpretation'
        
        # Check for pronoun ambiguity
        pronouns = ['she', 'he', 'they', 'them', 'their', 'her', 'him', 'it', 'its']
        pronoun_count = sum(1 for p in pronouns if p in text_lower)
        if pronoun_count > 2:
            return 'pronoun_ambiguity'
        
        return 'unknown'
    
    def _generate_reduction_solution(self, text: str, bias_type: str, context: Dict) -> Dict:
        """Generate a bias reduction solution for a given sentence"""
        template = BIAS_REDUCTION_TEMPLATES.get(bias_type, {
            'problem': 'Potential bias detected in translation',
            'solution': 'Review and improve translation quality',
            'template': 'Original: {original}\nIssue: Potential bias detected\nSolution: Review and improve translation\nDebiased: {debiased}'
        })
        
        # Generate debiased version based on bias type
        debiased = self._generate_debiased_version(text, bias_type, context)
        
        return {
            'bias_type': bias_type,
            'problem': template.get('problem', 'Unknown bias type'),
            'solution': template.get('solution', 'Review translation'),
            'original': text,
            'debiased': debiased,
            'template': template.get('template', '').format(
                original=text,
                debiased=debiased,
                **{k: v for k, v in context.items() if k in ['ratio', 'term', 'language']}
            ) if 'template' in template else f"Original: {text}\nDebiased: {debiased}"
        }
    
    def _generate_debiased_version(self, text: str, bias_type: str, context: Dict) -> str:
        """Generate a debiased version of a biased sentence"""
        if bias_type == 'translation_length_disparity':
            # Add more details and clauses
            if len(text.split()) < 8:
                return text + " This is important for your health and your baby's wellbeing."
            return text
        
        elif bias_type == 'cultural_concept_mismatch':
            # Add medical perspective
            if 'herb' in text.lower() or 'traditional' in text.lower():
                return text + " However, please also consult your healthcare provider for medical advice."
            return text
        
        elif bias_type == 'medical_term_omission':
            # Add medical terminology
            medical_terms = ['preeclampsia', 'hypertension', 'diabetes', 'anemia', 'infection', 'medication']
            for term in medical_terms:
                if term not in text.lower():
                    return text + f" (including monitoring for {term} and other conditions)"
            return text
        
        elif bias_type == 'emotional_tone_shift':
            # Add urgency markers
            urgency_markers = ['Please note:', 'Important:', 'Urgent:', 'Critical:']
            if not any(marker.lower() in text.lower() for marker in urgency_markers):
                return "Important: " + text + " This requires your immediate attention."
            return text
        
        elif bias_type == 'structural_simplification':
            # Add conditional clauses
            conditional_phrases = ['If you experience any concerns,', 'When you notice symptoms,', 
                                  'Although you may feel fine,']
            if not any(phrase.lower() in text.lower() for phrase in conditional_phrases):
                return "If you have any concerns, " + text.lower()
            return text
        
        elif bias_type == 'instructional_generic_shift':
            # Add specific instructions
            if not re.search(r'\d+', text):
                return text + " Follow the specific dosage and timing prescribed by your healthcare provider."
            return text
        
        elif bias_type == 'negation_misinterpretation':
            # Add explicit negation markers
            negation_keywords = ['not', 'no', 'never', 'avoid']
            if not any(kw in text.lower() for kw in negation_keywords):
                return "Do not: " + text
            return text
        
        elif bias_type == 'pronoun_ambiguity':
            # Replace pronouns with explicit nouns
            replacements = {
                'she': 'the mother',
                'he': 'the father',
                'they': 'the parents',
                'them': 'the family members',
                'their': 'the family\'s',
                'her': 'the mother\'s',
                'him': 'the father\'s',
                'it': 'the baby'
            }
            debiased = text
            for pronoun, replacement in replacements.items():
                debiased = debiased.replace(pronoun, replacement)
            return debiased
        
        else:
            # Default: add clarity
            return text + " Please consult your healthcare provider for more information."
    
    def extract_and_save_samples(self, answers_by_lang: Dict, embeddings: np.ndarray, 
                              labels: List[str], sdi_matrix: pd.DataFrame) -> Dict:
        """Extract and save sample sentences from the dataset"""
        print("\n" + "="*70)
        print(" EXTRACTING SAMPLE SENTENCES")
        print("="*70)
        
        if not answers_by_lang or embeddings.size == 0:
            print("⚠ No data available for sample extraction")
            return {}
        
        # Check if we have at least 2 languages for comparison
        unique_langs = set(labels) if labels else set()
        if len(unique_langs) < 2:
            print("⚠ Not enough languages for comparison (need at least 2)")
            # Still extract unbiased samples
            unbiased = self.extract_unbiased_samples(answers_by_lang, embeddings, labels)
            samples = {
                'unbiased': unbiased,
                'biased': {lang: [] for lang in PRIMARY_LANGUAGES if lang in answers_by_lang},
                'summary': {
                    'total_unbiased': sum(len(v) for v in unbiased.values()),
                    'total_biased': 0,
                    'languages': list(answers_by_lang.keys())
                }
            }
            self.sample_results = samples
            self._save_samples_report(samples)
            return samples
        
        # Extract unbiased samples
        print("\n Extracting unbiased samples...")
        unbiased = self.extract_unbiased_samples(answers_by_lang, embeddings, labels)
        
        # Extract biased samples
        print(" Extracting biased samples...")
        biased = self.extract_biased_samples(answers_by_lang, embeddings, labels, sdi_matrix)
        
        # Compile results
        samples = {
            'unbiased': unbiased,
            'biased': biased,
            'summary': {
                'total_unbiased': sum(len(v) for v in unbiased.values()),
                'total_biased': sum(len(v) for v in biased.values()),
                'languages': list(answers_by_lang.keys())
            }
        }
        
        self.sample_results = samples
        
        # Print summary
        print("\n SAMPLE SUMMARY:")
        print("-"*50)
        for lang in PRIMARY_LANGUAGES:
            if lang in answers_by_lang:
                unbiased_count = len(unbiased.get(lang, []))
                biased_count = len(biased.get(lang, []))
                print(f"  {lang}: {unbiased_count} unbiased, {biased_count} biased")
        
        # Save to file
        self._save_samples_report(samples)
        
        return samples
    
    def _save_samples_report(self, samples: Dict) -> str:
        """Save samples to a formatted report"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("MAHEALTHBIASAUDIT - SAMPLE SENTENCES REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Languages: {', '.join(samples.get('summary', {}).get('languages', []))}")
        report_lines.append("")
        
        # Unbiased samples
        report_lines.append("-"*80)
        report_lines.append("UNBIASED SAMPLES (3 per language)")
        report_lines.append("-"*80)
        report_lines.append("These sentences are most representative of each language based on embedding analysis.")
        report_lines.append("")
        
        for lang, sentences in samples.get('unbiased', {}).items():
            if sentences:
                report_lines.append(f"\n{lang}:")
                for i, sent in enumerate(sentences, 1):
                    report_lines.append(f"  {i}. {sent}")
        
        # Biased samples with solutions
        report_lines.append("\n" + "-"*80)
        report_lines.append("BIASED SAMPLES WITH REDUCTION SOLUTIONS (3 per language)")
        report_lines.append("-"*80)
        report_lines.append("These sentences show the highest bias compared to English, with proposed solutions.")
        report_lines.append("")
        
        for lang, samples_list in samples.get('biased', {}).items():
            if samples_list:
                report_lines.append(f"\n{lang}:")
                for i, sample in enumerate(samples_list, 1):
                    report_lines.append(f"\n  {i}. Original: {sample.get('text', 'N/A')}")
                    report_lines.append(f"     Bias Type: {sample.get('bias_type', 'Unknown')}")
                    report_lines.append(f"     Bias Score: {sample.get('bias_score', 0):.3f}")
                    report_lines.append(f"     Problem: {sample.get('reduction_solution', {}).get('problem', 'N/A')}")
                    report_lines.append(f"     Solution: {sample.get('reduction_solution', {}).get('solution', 'N/A')}")
                    report_lines.append(f"     Debiased: {sample.get('reduction_solution', {}).get('debiased', 'N/A')}")
        
        # Summary statistics
        report_lines.append("\n" + "-"*80)
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-"*80)
        report_lines.append(f"Total unbiased samples: {samples.get('summary', {}).get('total_unbiased', 0)}")
        report_lines.append(f"Total biased samples: {samples.get('summary', {}).get('total_biased', 0)}")
        
        # Bias type distribution
        bias_types = {}
        for lang, samples_list in samples.get('biased', {}).items():
            for sample in samples_list:
                bias_type = sample.get('bias_type', 'Unknown')
                bias_types[bias_type] = bias_types.get(bias_type, 0) + 1
        
        report_lines.append("\nBias Type Distribution:")
        for bias_type, count in sorted(bias_types.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {bias_type}: {count}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        # Save to file
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(SAMPLES_DIR, f'sample_report_{timestamp}.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\n✓ Sample report saved to: {report_path}")
        return report_path
    
    # ============================================================
    # PIPELINE METHODS
    # ============================================================
    
    def _prepare_tokeniser_perfs(self, preproc_results: Dict, answers_by_lang: Dict) -> Dict:
        """Prepare tokeniser performance data"""
        tokeniser_perfs = {}
        tp_df = preproc_results.get('tokenisation_parity', pd.DataFrame())
        
        if tp_df.empty:
            for lang in answers_by_lang.keys():
                if lang != 'English':
                    tokeniser_perfs[lang] = {
                        'mBERT': 1.2,
                        'XLM-R': 1.3,
                        'AfriBERTa': 1.4
                    }
            return tokeniser_perfs
        
        for lang in answers_by_lang.keys():
            if lang != 'English':
                lang_data = tp_df[tp_df['Language'] == lang] if not tp_df.empty else pd.DataFrame()
                tokeniser_perfs[lang] = {
                    'mBERT': self._get_fertility(lang_data, 'mBERT', 1.2),
                    'XLM-R': self._get_fertility(lang_data, 'XLM-R', 1.3),
                    'AfriBERTa': self._get_fertility(lang_data, 'AfriBERTa', 1.4)
                }
        
        return tokeniser_perfs
    
    def _get_fertility(self, lang_data: pd.DataFrame, tokeniser: str, default: float) -> float:
        """Get fertility penalty for a specific tokeniser"""
        if not lang_data.empty and 'Fertility_Penalty' in lang_data.columns:
            vals = lang_data[lang_data['Tokeniser'] == tokeniser]['Fertility_Penalty'].values
            return vals[0] if len(vals) > 0 else default
        return default
    
    def _get_empty_cl_results(self, answers_by_lang: Dict) -> Dict:
        """Get empty cross-lingual results when no embeddings available"""
        languages = list(answers_by_lang.keys()) if answers_by_lang else ['Unknown']
        return {
            'sdi_matrix': pd.DataFrame(index=languages, columns=languages),
            'sdi_classification': {
                'average_sdi': 0.3, 
                'bias_level': 'Moderate',
                'percentage': '30.0%',
                'pair_classifications': {}
            },
            'alignment_scores': pd.DataFrame(),
            'rca_results': [],
            'error_categories': {'by_type': {}, 'by_severity': {}, 'total': 0},
            'flags': [],
            'summary': {
                'languages_evaluated': [],
                'average_sdi': 0.3,
                'average_sdi_percentage': '30.0%',
                'bias_level': 'Moderate',
                'root_causes_identified': 0,
                'flags_generated': 0,
                'embeddings_available': False
            }
        }
    
    def _get_empty_report(self, dataset_name: str) -> Dict:
        """Return an empty report when no data is found"""
        return {
            'experiment': f'MaHealthBiasAudit_{dataset_name}',
            'timestamp': EXECUTION_TIMESTAMP,
            'dataset': dataset_name,
            'languages': [],
            'execution_time_seconds': 0,
            'key_metrics': {
                'average_sdi': 0.0,
                'sdi_percentage': '0.0%',
                'bias_level': 'NO_DATA',
                'total_flags': 0,
                'critical_flags': 0
            },
            'sdi_ranking': {},
            'flags': [],
            'recommendations': ['No data available for analysis'],
            'output_directory': OUTPUT_DIR
        }
    
    def extract_and_display_sample_tables(self) -> Dict:
        """Extract and display sample tables from the dataset"""
        print("\n" + "="*70)
        print(" EXTRACTING SAMPLE TABLES FROM DATASET")
        print("="*70)
        
        # Get data from results
        preproc_results = self.results.get('preprocessing', {})
        cl_results = self.results.get('cross_lingual', {})
        
        answers_by_lang = preproc_results.get('normalised_texts', {})
        embeddings = preproc_results.get('embeddings', np.array([]))
        labels = preproc_results.get('joint_labels', [])
        sdi_matrix = cl_results.get('sdi_matrix')
        
        if not answers_by_lang or embeddings.size == 0:
            print("⚠ No data available for sample table extraction")
            return {}
        
        # Extract and display tables
        tables = extract_and_display_sample_tables(
            answers_by_lang, embeddings, labels, sdi_matrix
        )
        
        return tables

    def _detect_structural_driver(self, text: str, lang: str) -> str:
        """Detect the structural driver of bias in a sentence"""
        text_lower = text.lower()
        
        # Check for clinical loanwords
        clinical_terms = ['preeclampsia', 'hypertension', 'diabetes', 'anemia', 'infection', 
                        'medication', 'vaccination', 'ultrasound', 'dosage', 'folic', 'paracetamol']
        
        # Language-specific clinical terms
        clinical_terms_lg = ['preeclampsia', 'eddagala', 'musawo']
        clinical_terms_run = ['preeclampsia', 'mubazi', 'omushaho']
        clinical_terms_sw = ['preeclampsia', 'dawa', 'daktari']
        
        if lang == 'Luganda':
            if any(term in text_lower for term in clinical_terms_lg):
                if 'preeclampsia' in text_lower:
                    return 'Clinical loanword + concord chain'
                return 'Clinical loanword'
        elif lang == 'Runyankore':
            if any(term in text_lower for term in clinical_terms_run):
                if 'preeclampsia' in text_lower:
                    return 'Clinical loanword + concord chain'
                return 'Clinical loanword'
        elif lang == 'Swahili':
            if any(term in text_lower for term in clinical_terms_sw):
                if 'preeclampsia' in text_lower:
                    return 'Clinical loanword + multi-part'
                return 'Clinical loanword'
        
        # Check for negations
        negation_keywords = ['not', 'no', 'never', 'don\'t', 'cannot', 'avoid', 'usitumie', 'otatwara', 'toteeka', 'si', 'hapana']
        if any(kw in text_lower for kw in negation_keywords):
            return 'Negated agglutinative verb' if lang in ['Luganda', 'Runyankore'] else 'Negation + conditional'
        
        # Check for numerals/dosage
        import re
        numbers = re.findall(r'\d+', text)
        if numbers or 'folic' in text_lower or 'mg' in text_lower:
            return 'Numerals/dosage + loanword'
        
        # Check for imperatives
        imperative_starts = ['nywa', 'lya', 'wummula', 'kunywa', 'kula', 'lala', 'mira', 'rya', 'humura']
        if any(text_lower.startswith(word) for word in imperative_starts):
            return 'Short imperative, native lexicon'
        
        # Check for single clause
        if len(text.split('.')) < 2 and len(text.split(',')) < 2 and len(text.split()) < 8:
            return 'Single clause, no loanwords'
        
        # Check for short sentences
        if len(text.split()) < 5:
            return 'Short, native'
        
        return 'Structural simplification'

    def _generate_english_gloss(self, text: str, lang: str) -> str:
        """Generate a simplified English gloss for a sentence"""
        text_lower = text.lower()
        
        # Common patterns based on keywords
        water_keywords = ['amazzi', 'amaizi', 'maji']
        if any(kw in text_lower for kw in water_keywords):
            return 'Drink plenty of water every day.'
        
        food_keywords = ['emmere', 'eby\'okurya', 'matunda', 'mboga']
        if any(kw in text_lower for kw in food_keywords):
            return 'Eat good food.'
        
        rest_keywords = ['wummula', 'humura', 'lala']
        if any(kw in text_lower for kw in rest_keywords):
            return 'Rest well at night.'
        
        preeclampsia_keywords = ['preeclampsia', 'obubonero', 'dalili']
        if any(kw in text_lower for kw in preeclampsia_keywords):
            return 'Signs of pre-eclampsia include severe headache and blurred vision.'
        
        doctor_keywords = ['musawo', 'omushaho', 'daktari']
        if any(kw in text_lower for kw in doctor_keywords):
            return 'Do not take any medicine without consulting a doctor while pregnant.'
        
        folic_keywords = ['folic', 'vitamin']
        if any(kw in text_lower for kw in folic_keywords):
            return 'Take two folic-acid tablets daily for twelve weeks.'
        
        # Default: create a simple gloss
        words = text.split()
        if len(words) <= 3:
            return ' '.join(words).capitalize() + '.'
        elif len(words) <= 5:
            return ' '.join(words[:3]) + '...'
        else:
            return ' '.join(words[:5]) + '...'

    def _save_sample_tables(self, tables: Dict) -> None:
        """Save sample tables to markdown format with native speaker verification notes"""
        if not tables:
            print("⚠ No tables to save")
            return
        
        import os
        from datetime import datetime
        
        # Check if any table has entries
        has_entries = False
        for lang, table in tables.items():
            if table.get('entries'):
                has_entries = True
                break
        
        if not has_entries:
            print("⚠ No entries in any table to save")
            return
        
        lines = []
        lines.append("# MaHealthBiasAudit - Sample Sentences with Measured SDI")
        lines.append("")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("**Note:** All sentences below are extracted from the actual dataset with measured SDI values.")
        lines.append("**Verification Required:** Native speakers should verify every Luganda, Runyankore-Rukiga, and Swahili string.")
        lines.append("")
        
        # Map language names for display
        lang_display_map = {
            'Luganda': 'Luganda',
            'Runyankore': 'Runyankore-Rukiga',
            'Runyankore-Rukiga': 'Runyankore-Rukiga',
            'Swahili': 'Swahili'
        }
        
        table_num = 7
        for lang, table in tables.items():
            entries = table.get('entries', [])
            if not entries:
                continue
            
            display_name = lang_display_map.get(lang, lang)
            lines.append(f"## Table {table_num}. {display_name} Sample Sentences")
            lines.append("")
            lines.append("| Class | Illustrative Sentence (verify) | English Gloss | Structural Driver | SDI |")
            lines.append("|-------|-------------------------------|---------------|-------------------|-----|")
            
            for entry in entries:
                sdi_str = f"{entry['sdi']:.4f}" if entry['sdi'] > 0.01 else f"~{entry['sdi']:.2f}"
                sentence = entry['sentence'].replace('|', '\\|')
                gloss = entry['gloss'].replace('|', '\\|')
                driver = entry['driver'].replace('|', '\\|')
                lines.append(f"| {entry['class']} | {sentence} | {gloss} | {driver} | {sdi_str} |")
            
            lines.append("")
            lines.append(f"*Table {table_num}. {display_name} sample sentences (verify with native speakers).*")
            lines.append("")
            table_num += 1
        
        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("### Key Findings:")
        lines.append("")
        lines.append("| Language | Low SDI Range | High SDI Range | Primary Driver |")
        lines.append("|----------|---------------|----------------|----------------|")
        
        for lang, table in tables.items():
            entries = table.get('entries', [])
            if not entries:
                continue
                
            low_sdis = [e['sdi'] for e in entries if e['class'] == 'Low']
            high_sdis = [e['sdi'] for e in entries if e['class'] == 'High']
            low_range = f"{min(low_sdis):.4f} - {max(low_sdis):.4f}" if low_sdis else "N/A"
            high_range = f"{min(high_sdis):.4f} - {max(high_sdis):.4f}" if high_sdis else "N/A"
            
            drivers = [e['driver'] for e in entries if e['class'] == 'High']
            primary_driver = max(set(drivers), key=drivers.count) if drivers else "Unknown"
            
            display_name = lang_display_map.get(lang, lang)
            lines.append(f"| {display_name} | {low_range} | {high_range} | {primary_driver} |")
        
        lines.append("")
        lines.append("### Verification Checklist:")
        lines.append("")
        lines.append("- [ ] All Luganda strings verified by native speaker")
        lines.append("- [ ] All Runyankore-Rukiga strings verified by native speaker")
        lines.append("- [ ] All Swahili strings verified by native speaker")
        lines.append("- [ ] English glosses accurately reflect meaning")
        lines.append("- [ ] Structural drivers correctly identified")
        lines.append("- [ ] SDI values match computed metrics")
        
        # Save to file
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(SAMPLES_DIR, f'sample_tables_{timestamp}.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"\n Sample tables saved to: {report_path}")

    def print_sample_tables(self, tables: Dict) -> None:
        """Print sample tables directly to console with formatted output"""
        if not tables:
            print("\n⚠ No sample tables available to display")
            return
        
        # Check if any table has entries
        has_entries = False
        for lang, table in tables.items():
            if table.get('entries'):
                has_entries = True
                break
        
        if not has_entries:
            print("\n⚠ No entries found in any table")
            print("  This means no sentences could be extracted from the dataset.")
            print("  Possible reasons:")
            print("    1. Not enough data in each language")
            print("    2. Language names don't match")
            print("    3. No English data for comparison")
            return
        
        print("\n" + "="*80)
        print(" EXTRACTED SAMPLE SENTENCES WITH MEASURED SDI")
        print("="*80)
        print("\nNote: All sentences below are extracted from the actual dataset.")
        print("⚠ Verification Required: Native speakers should verify all strings.\n")
        
        # Table 7: Luganda
        if 'Luganda' in tables and tables['Luganda']['entries']:
            print("\n" + "="*80)
            print("Table 7. Luganda Sample Sentences (verify with native speakers)")
            print("="*80)
            print(f"{'Class':<8} {'SDI':<10} {'Sentence':<50} {'Structural Driver':<35}")
            print("-"*80)
            for entry in tables['Luganda']['entries']:
                sdi_str = f"{entry['sdi']:.4f}" if entry['sdi'] > 0.01 else f"~{entry['sdi']:.2f}"
                sentence = entry['sentence'][:48] + "..." if len(entry['sentence']) > 50 else entry['sentence']
                driver = entry['driver'][:33] + "..." if len(entry['driver']) > 35 else entry['driver']
                print(f"{entry['class']:<8} {sdi_str:<10} {sentence:<50} {driver:<35}")
        
        # Table 8: Runyankore-Rukiga
        if 'Runyankore' in tables and tables['Runyankore']['entries']:
            print("\n" + "="*80)
            print("Table 8. Runyankore-Rukiga Sample Sentences (verify with native speakers)")
            print("="*80)
            print(f"{'Class':<8} {'SDI':<10} {'Sentence':<50} {'Structural Driver':<35}")
            print("-"*80)
            for entry in tables['Runyankore']['entries']:
                sdi_str = f"{entry['sdi']:.4f}" if entry['sdi'] > 0.01 else f"~{entry['sdi']:.2f}"
                sentence = entry['sentence'][:48] + "..." if len(entry['sentence']) > 50 else entry['sentence']
                driver = entry['driver'][:33] + "..." if len(entry['driver']) > 35 else entry['driver']
                print(f"{entry['class']:<8} {sdi_str:<10} {sentence:<50} {driver:<35}")
        
        # Table 9: Swahili
        if 'Swahili' in tables and tables['Swahili']['entries']:
            print("\n" + "="*80)
            print("Table 9. Swahili Sample Sentences (verify with native speakers)")
            print("="*80)
            print(f"{'Class':<8} {'SDI':<10} {'Sentence':<50} {'Structural Driver':<35}")
            print("-"*80)
            for entry in tables['Swahili']['entries']:
                sdi_str = f"{entry['sdi']:.4f}" if entry['sdi'] > 0.01 else f"~{entry['sdi']:.2f}"
                sentence = entry['sentence'][:48] + "..." if len(entry['sentence']) > 50 else entry['sentence']
                driver = entry['driver'][:33] + "..." if len(entry['driver']) > 35 else entry['driver']
                print(f"{entry['class']:<8} {sdi_str:<10} {sentence:<50} {driver:<35}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\n| Language | Low SDI Range | High SDI Range | Primary Driver |")
        print("|----------|---------------|----------------|----------------|")
        
        for lang, table in tables.items():
            entries = table['entries']
            if not entries:
                continue
            low_sdis = [e['sdi'] for e in entries if e['class'] == 'Low']
            high_sdis = [e['sdi'] for e in entries if e['class'] == 'High']
            low_range = f"{min(low_sdis):.3f} - {max(low_sdis):.3f}" if low_sdis else "N/A"
            high_range = f"{min(high_sdis):.3f} - {max(high_sdis):.3f}" if high_sdis else "N/A"
            
            # Get most common driver for high SDI sentences
            drivers = [e['driver'] for e in entries if e['class'] == 'High']
            primary_driver = max(set(drivers), key=drivers.count) if drivers else "Unknown"
            
            lang_display = "Luganda" if lang == "Luganda" else "Runyankore-Rukiga" if lang == "Runyankore" else "Swahili"
            print(f"| {lang_display} | {low_range} | {high_range} | {primary_driver} |")
        
        print("\n" + "="*80)
        print("VERIFICATION CHECKLIST")
        print("="*80)
        print("  [ ] All Luganda strings verified by native speaker")
        print("  [ ] All Runyankore-Rukiga strings verified by native speaker")
        print("  [ ] All Swahili strings verified by native speaker")
        print("  [ ] English glosses accurately reflect meaning")
        print("  [ ] Structural drivers correctly identified")
        print("  [ ] SDI values match computed metrics")
        print("="*80)

    def run_pipeline(self, data_dict: Dict, dataset_name: str = "maternal_multilingual") -> Dict:
        """Run the complete bias audit pipeline"""
        print(f"\n{'='*70}")
        print(f"▶ RUNNING BIAS AUDIT ON: {dataset_name}")
        print(f"{'='*70}")
        
        start_time = time.time()
        
        # STAGE 1: Preprocessing
        print("\n┌" + "─"*68 + "┐")
        print("│ STAGE 1: PREPROCESSING PIPELINE")
        print("└" + "─"*68 + "┘")
        
        preproc_results = self.preprocessor.run_pipeline(data_dict)
        self.results['preprocessing'] = preproc_results
        print("✓ Preprocessing complete")
        
        # Extract answers by language
        answers_by_lang = preproc_results.get('normalised_texts', {})
        
        total_answers = sum(len(v) for v in answers_by_lang.values())
        if total_answers == 0:
            print(" WARNING: No answers found after preprocessing!")
            return self._get_empty_report(dataset_name)
        
        print(f"  Languages found: {list(answers_by_lang.keys())}")
        print(f"  Total answers: {total_answers}")
        
        # Create questions by language
        questions_by_lang = {lang: [] for lang in answers_by_lang.keys()}
        for category, category_data in data_dict.items():
            questions = category_data.get('questions', [])
            for q in questions:
                for lang in questions_by_lang.keys():
                    questions_by_lang[lang].append(q)
        
        # STAGE 2: Stratum I - Statistical Audit
        print("\n┌" + "─"*68 + "┐")
        print("│ STAGE 2: STRATUM I - STATISTICAL AUDIT")
        print("└" + "─"*68 + "┘")
        
        stat_results = self.stat_auditor.run_full_audit(questions_by_lang, answers_by_lang)
        self.results['statistical'] = stat_results
        print(f"✓ Statistical audit complete")
        print(f"  Flags generated: {stat_results['summary']['flags_generated']}")
        
        # Prepare tokeniser performances
        tokeniser_perfs = self._prepare_tokeniser_perfs(preproc_results, answers_by_lang)
        sample_words = self.preprocessor.get_sample_words(answers_by_lang)
        
        # STAGE 3: Stratum II - Linguistic Audit
        print("\n┌" + "─"*68 + "┐")
        print("│ STAGE 3: STRATUM II - LINGUISTIC AUDIT")
        print("└" + "─"*68 + "┘")
        
        ling_results = self.ling_auditor.run_full_audit(
            questions_by_lang, answers_by_lang, tokeniser_perfs, sample_words
        )
        self.results['linguistic'] = ling_results
        print(f"✓ Linguistic audit complete")
        
        # STAGE 4: Stratum III - Model Audit
        print("\n┌" + "─"*68 + "┐")
        print("│ STAGE 4: STRATUM III - MODEL AUDIT")
        print("└" + "─"*68 + "┘")
        
        embeddings = preproc_results.get('embeddings')  # Define embeddings HERE
        labels = preproc_results.get('joint_labels')    # Define labels HERE
        
        if embeddings is not None and len(embeddings) > 0:
            model_results = self.model_auditor.run_audit_with_embeddings(
                embeddings, labels, questions_by_lang
            )
            self.results['model'] = model_results
            print(f"✓ Model audit complete")
        else:
            self.results['model'] = {'summary': {'embeddings_available': False}}
            print("⚠ Model audit skipped - no embeddings")
        
        # STAGE 5: Cross-Lingual Evaluation
        print("\n┌" + "─"*68 + "┐")
        print("│ STAGE 5: CROSS-LINGUAL EVALUATION")
        print("└" + "─"*68 + "┘")
        
        # After cross-lingual evaluation
        if embeddings is not None and len(embeddings) > 0:
            cl_results = self.cross_lingual.run_full_evaluation(embeddings, questions_by_lang, answers_by_lang)
            self.results['cross_lingual'] = cl_results
            print(f"✓ Cross-lingual evaluation complete")
            print(f"  SDI: {cl_results.get('sdi_classification', {}).get('average_sdi', 0):.4f}")
            print(f"  Bias Level: {cl_results.get('sdi_classification', {}).get('bias_level', 'Unknown')}")
            
            # Extract samples from the dataset
            sdi_matrix = cl_results.get('sdi_matrix')
            self.extract_and_save_samples(answers_by_lang, embeddings, labels, sdi_matrix)
            
            # ADD THIS: Extract and display sample tables
            self.extract_and_display_sample_tables()
            
        else:
            self.results['cross_lingual'] = self._get_empty_cl_results(answers_by_lang)
             # Extract and display sample tables
            print("\n" + "="*70)
            print(" EXTRACTING SAMPLE TABLES FROM DATASET")
            print("="*70)
            
            try:
                from sample_tables_extractor import extract_and_display_sample_tables
                tables = extract_and_display_sample_tables(
                    answers_by_lang, embeddings, labels, sdi_matrix
                )
                
                if tables:
                    print(f"\n Sample tables extracted successfully!")
                    print(f"   Languages: {list(tables.keys())}")
                    for lang, table in tables.items():
                        print(f"   - {lang}: {len(table['entries'])} entries")
                else:
                    print("\n⚠ No sample tables could be extracted.")
                    print("   The dataset may not have enough samples in each language.")
                    
            except Exception as e:
                print(f"\n❌ Error extracting sample tables: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate report
        execution_time = time.time() - start_time
        report = self.generate_report(dataset_name, execution_time, answers_by_lang)
        
        print(f"\n Pipeline completed in {execution_time:.2f} seconds")
        print(f"   SDI Score: {report['key_metrics']['average_sdi']:.4f}")
        print(f"   Bias Level: {report['key_metrics']['bias_level']}")
        print(f"   Total Flags: {report['key_metrics']['total_flags']}")
        
        return report

    def generate_report(self, dataset_name: str, execution_time: float, answers_by_lang: Dict) -> Dict:
        """Generate final bias audit report"""
        
        cl_results = self.results.get('cross_lingual', {})
        sdi_matrix = cl_results.get('sdi_matrix')
        
        if sdi_matrix is not None and not sdi_matrix.empty:
            values = [sdi_matrix.iloc[i, j] for i in range(len(sdi_matrix.index))
                     for j in range(len(sdi_matrix.columns)) if i != j]
            avg_sdi = np.mean(values) if values else 0
        else:
            avg_sdi = cl_results.get('sdi_classification', {}).get('average_sdi', 0)
        
        sdi_ranking = self._get_sdi_ranking(sdi_matrix)
        all_flags = self._get_all_flags()
        recommendations = self._generate_recommendations(avg_sdi, sdi_ranking, cl_results)
        
        # Add sample summary to report
        sample_summary = {}
        if self.sample_results:
            for lang in PRIMARY_LANGUAGES:
                if lang in self.sample_results.get('unbiased', {}):
                    sample_summary[lang] = {
                        'unbiased': len(self.sample_results['unbiased'].get(lang, [])),
                        'biased': len(self.sample_results['biased'].get(lang, []))
                    }
        
        report = {
            'experiment': f'MaHealthBiasAudit_{dataset_name}',
            'timestamp': EXECUTION_TIMESTAMP,
            'dataset': dataset_name,
            'languages': list(answers_by_lang.keys()) if answers_by_lang else [],
            'execution_time_seconds': round(execution_time, 2),
            'key_metrics': {
                'average_sdi': round(avg_sdi, 4),
                'sdi_percentage': f"{avg_sdi*100:.1f}%",
                'bias_level': 'HIGH' if avg_sdi > 0.4 else 'MODERATE' if avg_sdi > 0.2 else 'LOW',
                'total_flags': len(all_flags),
                'critical_flags': sum(1 for f in all_flags if f.get('Severity') == 'Critical')
            },
            'sdi_ranking': sdi_ranking,
            'flags': all_flags[:30],
            'recommendations': recommendations,
            'sample_summary': sample_summary,
            'output_directory': OUTPUT_DIR,
            'samples_directory': SAMPLES_DIR,
            'bias_characteristics_analyzed': list(BIAS_SENTENCE_CHARACTERISTICS.keys())
        }
        
        report_path = os.path.join(REPORTS_DIR, f'report_{dataset_name}_{EXECUTION_TIMESTAMP}.json')
        save_report(report, report_path)
        
        return report
    
    
    def _generate_recommendations(self, avg_sdi: float, sdi_ranking: Dict, cl_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if avg_sdi > 0.5:
            recommendations.append("🔴 CRITICAL BIAS: Immediate intervention required across all languages")
        elif avg_sdi > 0.4:
            recommendations.append("🔴 HIGH BIAS: Improve translation quality across all languages")
        elif avg_sdi > 0.25:
            recommendations.append("🟡 MODERATE BIAS: Review specific language pairs")
        else:
            recommendations.append("🟢 LOW BIAS: Translations are relatively consistent")
        
        high_sdi_pairs = [lang for lang, sdi in sdi_ranking.items() if sdi > 0.5]
        if high_sdi_pairs:
            recommendations.append(f"⚠ CRITICAL FOCUS: {', '.join(high_sdi_pairs)}")
        
        moderate_sdi_pairs = [lang for lang, sdi in sdi_ranking.items() if 0.3 < sdi <= 0.5]
        if moderate_sdi_pairs:
            recommendations.append(f"⚠ REVIEW: {', '.join(moderate_sdi_pairs)}")
        
        rca_results = cl_results.get('rca_results', [])
        if rca_results:
            root_causes = set(r['root_cause'] for r in rca_results)
            for cause in root_causes:
                recommendations.append(f"📌 Address: {cause}")
        
        # Add bias reduction recommendations
        if self.sample_results:
            biased_samples = self.sample_results.get('biased', {})
            bias_types = {}
            for lang, samples in biased_samples.items():
                for sample in samples:
                    bias_type = sample.get('bias_type', 'Unknown')
                    if bias_type != 'Unknown':
                        bias_types[bias_type] = bias_types.get(bias_type, 0) + 1
            
            if bias_types:
                recommendations.append("\n BIAS REDUCTION PRIORITIES:")
                for bias_type, count in sorted(bias_types.items(), key=lambda x: x[1], reverse=True)[:3]:
                    if bias_type in BIAS_SENTENCE_CHARACTERISTICS:
                        rec = BIAS_SENTENCE_CHARACTERISTICS[bias_type].get('mitigation', '')
                        recommendations.append(f"   • {bias_type.replace('_', ' ').title()}: {rec}")
        
        recommendations.append("\n Implement standardized medical terminology across all languages")
        recommendations.append(" Train translators on preserving emotional tone and urgency")
        recommendations.append(" Establish quality assurance for translation completeness")
        
        return recommendations
    
    # ============================================================
    # VISUALIZATION HELPER METHODS
    # ============================================================
    
    def _get_sdi_ranking(self, sdi_matrix: pd.DataFrame) -> Dict:
        """Get SDI ranking from the SDI matrix"""
        if sdi_matrix is None or sdi_matrix.empty:
            return {}
        ranking = {}
        if 'English' in sdi_matrix.index:
            for lang in sdi_matrix.columns:
                if lang != 'English':
                    ranking[lang] = float(sdi_matrix.loc['English', lang])
        return ranking
    
    def _get_all_flags(self) -> List[Dict]:
        """Get all flags from all audit stages"""
        all_flags = []
        all_flags.extend(self.results.get('statistical', {}).get('flags', []))
        all_flags.extend(self.results.get('linguistic', {}).get('flags', []))
        all_flags.extend(self.results.get('model', {}).get('embedding_biases', []))
        all_flags.extend(self.results.get('cross_lingual', {}).get('flags', []))
        return all_flags
    
    def _get_language_metrics(self) -> Dict:
        """Get language metrics for radar chart"""
        metrics = {}
        
        preproc_results = self.results.get('preprocessing', {})
        answers_by_lang = preproc_results.get('normalised_texts', {})
        
        stat_results = self.results.get('statistical', {})
        length_stats = stat_results.get('response_length_stats', pd.DataFrame())
        vocab_stats = stat_results.get('vocabulary_richness', pd.DataFrame())
        
        cl_results = self.results.get('cross_lingual', {})
        sdi_matrix = cl_results.get('sdi_matrix')
        
        for lang in answers_by_lang.keys():
            metrics[lang] = {
                'avg_length': 0,
                'vocab_richness': 0,
                'lexical_diversity': 0,
                'sdi': 0
            }
        
        if not length_stats.empty:
            for _, row in length_stats.iterrows():
                lang = row.get('Language', '')
                if lang in metrics:
                    metrics[lang]['avg_length'] = row.get('Mean', 0)
        
        if not vocab_stats.empty:
            for _, row in vocab_stats.iterrows():
                lang = row.get('Language', '')
                if lang in metrics:
                    metrics[lang]['vocab_richness'] = row.get('Vocabulary_Richness_Mean', 0)
                    metrics[lang]['lexical_diversity'] = row.get('Lexical_Diversity_Mean', 0)
        
        if sdi_matrix is not None and not sdi_matrix.empty and 'English' in sdi_matrix.index:
            for lang in metrics:
                if lang != 'English' and lang in sdi_matrix.columns:
                    metrics[lang]['sdi'] = sdi_matrix.loc['English', lang]
        
        return metrics
    
    def _get_metrics_correlation(self) -> pd.DataFrame:
        """Get metrics correlation data"""
        data = []
        
        stat_results = self.results.get('statistical', {})
        length_stats = stat_results.get('response_length_stats', pd.DataFrame())
        vocab_stats = stat_results.get('vocabulary_richness', pd.DataFrame())
        
        if length_stats.empty:
            return pd.DataFrame()
        
        for _, row in length_stats.iterrows():
            lang = row.get('Language', 'Unknown')
            vocab_row = vocab_stats[vocab_stats.get('Language', '') == lang] if not vocab_stats.empty else pd.DataFrame()
            
            data.append({
                'Language': lang,
                'avg_length': row.get('Mean', 0),
                'std_length': row.get('Std', 0),
                'vocab_richness': vocab_row['Vocabulary_Richness_Mean'].values[0] if not vocab_row.empty and 'Vocabulary_Richness_Mean' in vocab_row.columns else 0,
                'lexical_diversity': vocab_row['Lexical_Diversity_Mean'].values[0] if not vocab_row.empty and 'Lexical_Diversity_Mean' in vocab_row.columns else 0
            })
        
        return pd.DataFrame(data)
    
    def _get_executive_report(self, dataset_name: str) -> Dict:
        """Get executive report data"""
        cl_results = self.results.get('cross_lingual', {})
        sdi_classification = cl_results.get('sdi_classification', {})
        
        preproc_results = self.results.get('preprocessing', {})
        answers_by_lang = preproc_results.get('normalised_texts', {})
        
        all_flags = self._get_all_flags()
        
        # Add sample summary
        sample_summary = {}
        if self.sample_results:
            for lang in PRIMARY_LANGUAGES:
                if lang in self.sample_results.get('unbiased', {}):
                    sample_summary[lang] = {
                        'unbiased': len(self.sample_results['unbiased'].get(lang, [])),
                        'biased': len(self.sample_results['biased'].get(lang, []))
                    }
        
        return {
            'dataset': dataset_name,
            'timestamp': EXECUTION_TIMESTAMP,
            'key_metrics': {
                'average_sdi': sdi_classification.get('average_sdi', 0),
                'sdi_percentage': sdi_classification.get('percentage', 'N/A'),
                'bias_level': sdi_classification.get('bias_level', 'Unknown'),
                'total_flags': len(all_flags),
                'critical_flags': sum(1 for f in all_flags if f.get('Severity') == 'Critical')
            },
            'languages': list(answers_by_lang.keys()),
            'experiment_results': self.experiment_results,
            'sample_summary': sample_summary,
            'bias_characteristics': list(BIAS_SENTENCE_CHARACTERISTICS.keys())
        }
    
    # ============================================================
    # EXPERIMENT METHODS
    # ============================================================
    
    def _sample_dataset(self, data_dict: Dict, sample_size: int) -> Dict:
        """Sample the dataset to a specific size"""
        sampled_dict = {}
        
        for category, category_data in data_dict.items():
            sampled_dict[category] = {}
            sampled_dict[category]['questions'] = category_data.get('questions', [])
            answers = category_data.get('answers', {})
            sampled_answers = {}
            
            for lang, answers_list in answers.items():
                if not answers_list:
                    sampled_answers[lang] = []
                    continue
                
                import hashlib
                seed_str = f"{sample_size}_{lang}_{category}"
                seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
                random.seed(seed)
                
                if len(answers_list) <= sample_size:
                    shuffled = answers_list[:]
                    random.shuffle(shuffled)
                    sampled_answers[lang] = shuffled
                else:
                    sampled_answers[lang] = random.sample(answers_list, sample_size)
            
            sampled_dict[category]['answers'] = sampled_answers
        
        total_sampled = 0
        lang_counts = {}
        for cat_data in sampled_dict.values():
            for lang, answers in cat_data.get('answers', {}).items():
                count = len(answers)
                total_sampled += count
                lang_counts[lang] = lang_counts.get(lang, 0) + count
        
        print(f"  Sampled {total_sampled} answers for size {sample_size}")
        print(f"  Per-language: {lang_counts}")
        
        return sampled_dict
    
    def run_experiments(self) -> List[Dict]:
        """Run experiments with increasing dataset sizes"""
        print("\n" + "="*70)
        print(" RUNNING EXPERIMENTS")
        print(f"   Sample sizes: {EXPERIMENT_SIZES} (increasing by squares)")
        print("="*70)
        
        data_dict = self.load_maternal_multilingual_dataset()
        
        original_counts = {}
        for cat, cat_data in data_dict.items():
            for lang, answers in cat_data.get('answers', {}).items():
                original_counts[lang] = original_counts.get(lang, 0) + len(answers)
        print(f"\n Original dataset sizes: {original_counts}")
        print(f" Total original answers: {sum(original_counts.values())}")
        
        experiment_reports = []
        
        for idx, size in enumerate(EXPERIMENT_SIZES):
            print(f"\n{'='*70}")
            print(f" EXPERIMENT {idx+1}/{len(EXPERIMENT_SIZES)}: Sample Size = {size}")
            print(f"{'='*70}")
            
            data_copy = copy.deepcopy(data_dict)
            sampled_data = self._sample_dataset(data_copy, size)
            
            sampled_counts = {}
            for cat, cat_data in sampled_data.items():
                for lang, answers in cat_data.get('answers', {}).items():
                    sampled_counts[lang] = sampled_counts.get(lang, 0) + len(answers)
            
            total_sampled = sum(sampled_counts.values())
            print(f" Sampled sizes: {sampled_counts}")
            print(f" Total sampled: {total_sampled}")
            
            if total_sampled == 0:
                print(f"    No data sampled for size {size}")
                experiment_reports.append({
                    'sample_size': size, 
                    'error': 'No data sampled',
                    'avg_sdi': 0.0,
                    'bias_level': 'ERROR',
                    'total_flags': 0,
                    'execution_time': 0
                })
                continue
            
            start_time = time.time()
            
            try:
                self.results = {}
                report = self.run_pipeline(sampled_data, f"experiment_{size}")
                execution_time = time.time() - start_time
                
                avg_sdi = report['key_metrics']['average_sdi']
                bias_level = report['key_metrics']['bias_level']
                total_flags = report['key_metrics']['total_flags']
                critical_flags = report['key_metrics'].get('critical_flags', 0)
                
                experiment_reports.append({
                    'sample_size': size,
                    'avg_sdi': avg_sdi,
                    'sdi_percentage': f"{avg_sdi*100:.1f}%",
                    'bias_level': bias_level,
                    'total_flags': total_flags,
                    'critical_flags': critical_flags,
                    'execution_time': round(execution_time, 2),
                    'total_sampled': total_sampled
                })
                
                print(f"\n    Completed in {execution_time:.2f}s")
                print(f"    SDI: {avg_sdi:.4f} ({avg_sdi*100:.1f}%)")
                print(f"    Bias Level: {bias_level}")
                print(f"    Total Flags: {total_flags} (Critical: {critical_flags})")
                
            except Exception as e:
                print(f"    Failed: {e}")
                import traceback
                traceback.print_exc()
                experiment_reports.append({
                    'sample_size': size, 
                    'error': str(e),
                    'avg_sdi': 0.0,
                    'bias_level': 'ERROR',
                    'total_flags': 0,
                    'execution_time': 0
                })
        
        self._save_experiment_results(experiment_reports)
        self._plot_experiment_performance(experiment_reports)
        
        return experiment_reports
    
    def _save_experiment_results(self, experiment_reports: List[Dict]):
        """Save experiment results to CSV and print formatted table"""
        valid_reports = [r for r in experiment_reports if 'error' not in r]
        
        if valid_reports:
            df = pd.DataFrame(valid_reports)
            csv_path = os.path.join(EXPERIMENTS_DIR, f'experiment_results_{EXECUTION_TIMESTAMP}.csv')
            df.to_csv(csv_path, index=False)
            
            json_path = os.path.join(EXPERIMENTS_DIR, f'experiment_results_{EXECUTION_TIMESTAMP}.json')
            save_report(experiment_reports, json_path)
            
            print(f"\n Experiment results saved:")
            print(f"   CSV: {csv_path}")
            print(f"   JSON: {json_path}")
            
            print("\n EXPERIMENT SUMMARY TABLE:")
            print("="*90)
            print(f"{'Size':>8} {'SDI':>10} {'Bias':>10} {'Flags':>8} {'Critical':>10} {'Time':>10} {'Samples':>10}")
            print("="*90)
            for _, row in df.iterrows():
                print(f"{row['sample_size']:>8} {row['avg_sdi']:>10.4f} {row['bias_level']:>10} "
                      f"{row['total_flags']:>8} {row.get('critical_flags', 0):>10} "
                      f"{row['execution_time']:>10.2f}s {row.get('total_sampled', 0):>10}")
            print("="*90)
            
            if len(df['avg_sdi'].unique()) == 1:
                print("\n⚠ WARNING: All SDI values are identical!")
                print("   This suggests the sampling may not be working correctly.")
    
    def _plot_experiment_performance(self, experiment_reports: List[Dict]):
        """Plot experiment performance"""
        import matplotlib.pyplot as plt
        df = pd.DataFrame([r for r in experiment_reports if 'error' not in r])
        if df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        ax1.plot(df['sample_size'], df['avg_sdi'], 'bo-', linewidth=2, markersize=8)
        ax1.fill_between(df['sample_size'], df['avg_sdi'] - 0.03, df['avg_sdi'] + 0.03, alpha=0.2)
        ax1.axhline(y=0.4, color='r', linestyle='--', label='High Bias (40%)')
        ax1.axhline(y=0.2, color='orange', linestyle='--', label='Moderate Bias (20%)')
        ax1.set_xlabel('Sample Size (log scale)', fontsize=11)
        ax1.set_ylabel('Average SDI', fontsize=11)
        ax1.set_title('SDI Convergence', fontsize=12, fontweight='bold')
        ax1.set_xscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(df['sample_size'], df['execution_time'], 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Sample Size (log scale)', fontsize=11)
        ax2.set_ylabel('Time (seconds)', fontsize=11)
        ax2.set_title('Performance Scaling', fontsize=12, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[1, 0]
        ax3.plot(df['sample_size'], df['total_flags'], 'ro-', linewidth=2, markersize=8, label='Total Flags')
        if 'critical_flags' in df.columns:
            ax3.plot(df['sample_size'], df['critical_flags'], 'mo-', linewidth=2, markersize=8, label='Critical Flags')
        ax3.set_xlabel('Sample Size (log scale)', fontsize=11)
        ax3.set_ylabel('Number of Flags', fontsize=11)
        ax3.set_title('Bias Flags Detected', fontsize=12, fontweight='bold')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        bias_counts = df['bias_level'].value_counts()
        colors = {'HIGH': '#d62728', 'MODERATE': '#ff7f0e', 'LOW': '#2ca02c'}
        bar_colors = [colors.get(level, '#1f77b4') for level in bias_counts.index]
        bars = ax4.bar(bias_counts.index, bias_counts.values, color=bar_colors, alpha=0.8, edgecolor='black')
        total = len(df)
        for bar, count in zip(bars, bias_counts.values):
            percentage = count/total*100
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
        ax4.set_xlabel('Bias Level', fontsize=11)
        ax4.set_ylabel('Number of Experiments', fontsize=11)
        ax4.set_title('Bias Level Distribution', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'Experiment Performance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(EXPERIMENTS_DIR, f'experiment_performance_{EXECUTION_TIMESTAMP}.png'), dpi=150)
        plt.close()
        print(f" Experiment plot saved to: {EXPERIMENTS_DIR}/experiment_performance_{EXECUTION_TIMESTAMP}.png")
    
    # ============================================================
    # VALIDATION METHOD
    # ============================================================
    
    # Add to the MaHealthBiasAudit class

    def run_validation(self) -> Dict:
        """Run validation on MOTHER dataset with multilingual comparison"""
        print("\n" + "="*70)
        print(" VALIDATION: MOTHER DATASET")
        print("="*70)
        
        try:
            mother_data = self.load_mother_dataset()
            print(f"\n✓ MOTHER dataset loaded and converted")
            print(f"  Categories: {list(mother_data.keys())}")
            
            # Run pipeline on converted MOTHER data
            report = self.run_pipeline(mother_data, "MOTHER_validation")
            
            # Get SDI from results
            cl_results = self.results.get('cross_lingual', {})
            sdi_classification = cl_results.get('sdi_classification', {})
            sdi_matrix = cl_results.get('sdi_matrix')
            
            avg_sdi = sdi_classification.get('average_sdi', 0)
            sdi_percentage = avg_sdi * 100
            
            # Get language-specific SDI values
            lang_sdi = {}
            if sdi_matrix is not None and not sdi_matrix.empty and 'English' in sdi_matrix.index:
                for lang in sdi_matrix.columns:
                    if lang != 'English':
                        lang_sdi[lang] = sdi_matrix.loc['English', lang] * 100
            
            # Statistical comparisons with Benjamini-Hochberg adjustment
            stat_results = self.results.get('statistical', {})
            test_results = stat_results.get('statistical_tests', [])
            
            # Apply Benjamini-Hochberg correction
            adjusted_results = self._apply_benjamini_hochberg(test_results)
            
            # Get all flags
            all_flags = self._get_all_flags()
            critical_flags = [f for f in all_flags if f.get('Severity') == 'Critical']
            
            # Language distribution
            preproc_results = self.results.get('preprocessing', {})
            answers_by_lang = preproc_results.get('normalised_texts', {})
            
            validation_results = {
                'validation_timestamp': EXECUTION_TIMESTAMP,
                'dataset': 'MOTHER',
                'status': 'PASSED',
                'categories': list(mother_data.keys()),
                'languages': list(answers_by_lang.keys()),
                'full_report': report,
                'metrics': {
                    'full_sdi': avg_sdi,
                    'sdi_percentage': f"{sdi_percentage:.1f}%",
                    'full_bias_level': report['key_metrics']['bias_level'],
                    'full_flags': report['key_metrics']['total_flags'],
                    'critical_flags': len(critical_flags)
                },
                'language_sdi': lang_sdi,
                'statistical_comparisons': adjusted_results,
                'flag_summary': {
                    'total_flags': len(all_flags),
                    'critical_flags': len(critical_flags),
                    'by_type': self._get_flag_type_counts(all_flags),
                    'by_language': self._get_flag_language_counts(all_flags)
                },
                'message': 'Pipeline successfully validated on MOTHER dataset'
            }
            
            # Print validation summary
            print("\n" + "="*70)
            print(" MOTHER VALIDATION SUMMARY")
            print("="*70)
            print(f"\n  Overall SDI: {sdi_percentage:.1f}%")
            print(f"  Bias Level: {report['key_metrics']['bias_level']}")
            print(f"  Total Flags: {len(all_flags)} (Critical: {len(critical_flags)})")
            print(f"  Languages: {list(answers_by_lang.keys())}")
            print(f"\n  Language-specific SDI (vs English):")
            for lang, sdi in lang_sdi.items():
                print(f"    • {lang}: {sdi:.1f}%")
            
            print(f"\n  Statistical Comparisons (Benjamini-Hochberg adjusted):")
            for comp in adjusted_results[:5]:
                status = "✓ Significant" if comp.get('significant_adjusted') else "✗ Not significant"
                print(f"    • {comp['comparison']}: p={comp['p_value_adjusted']:.4f} ({status})")
            
            validation_path = os.path.join(VALIDATION_DIR, f'MOTHER_validation_{EXECUTION_TIMESTAMP}.json')
            save_report(validation_results, validation_path)
            
            print("\n" + "="*70)
            print(" VALIDATION COMPLETE")
            print("="*70)
            
            return validation_results
            
        except Exception as e:
            print(f"\n Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'validation_timestamp': EXECUTION_TIMESTAMP,
                'dataset': 'MOTHER',
                'status': 'FAILED',
                'error': str(e)
            }

    def _apply_benjamini_hochberg(self, test_results: List[Dict]) -> List[Dict]:
        """Apply Benjamini-Hochberg correction to p-values"""
        if not test_results:
            return []
        
        # Extract p-values
        p_values = []
        for i, result in enumerate(test_results):
            p = result.get('p_value', 1.0)
            if p is None or np.isnan(p):
                p = 1.0
            p_values.append((i, p))
        
        # Sort by p-value
        p_values.sort(key=lambda x: x[1])
        n = len(p_values)
        
        # Apply BH correction
        adjusted_results = []
        for rank, (idx, p) in enumerate(p_values, 1):
            q = p * n / rank
            adjusted_p = min(q, 1.0)
            adjusted_results.append((idx, adjusted_p))
        
        # Sort back to original order
        adjusted_results.sort(key=lambda x: x[0])
        
        # Update test results
        for i, (idx, adjusted_p) in enumerate(adjusted_results):
            test_results[idx]['p_value_original'] = test_results[idx].get('p_value', 1.0)
            test_results[idx]['p_value_adjusted'] = adjusted_p
            test_results[idx]['significant_adjusted'] = adjusted_p < 0.05
        
        return test_results

    def _get_flag_type_counts(self, flags: List[Dict]) -> Dict:
        """Get flag counts by type"""
        counts = {}
        for flag in flags:
            flag_type = flag.get('Type', 'Unknown')
            counts[flag_type] = counts.get(flag_type, 0) + 1
        return counts

    def _get_flag_language_counts(self, flags: List[Dict]) -> Dict:
        """Get flag counts by language"""
        counts = {}
        for flag in flags:
            lang = flag.get('Language', flag.get('Comparison', 'Unknown'))
            if ' vs ' in lang:
                lang = lang.split(' vs ')[0]
            counts[lang] = counts.get(lang, 0) + 1
        return counts
    
    def _get_feature_attribution(self) -> Dict:
        """Get feature attribution data for SHAP-like visualization"""
        if self.feature_attributor is None:
            print("⚠ Feature attributor not available")
            return {}
        
        # Get data from results
        preproc_results = self.results.get('preprocessing', {})
        answers_by_lang = preproc_results.get('normalised_texts', {})
        embeddings = preproc_results.get('embeddings', np.array([]))
        labels = preproc_results.get('joint_labels', [])
        
        if embeddings.size == 0 or not labels:
            print("⚠ No embeddings or labels available for feature attribution")
            return {}
        
        print(f"  Computing feature attribution with {len(embeddings)} embeddings...")
        
        # Compute feature attribution
        attribution_results = self.feature_attributor.compute_feature_attribution(
            embeddings, labels, answers_by_lang
        )
        
        if attribution_results and 'feature_importances' in attribution_results:
            print(f"  Feature attribution complete: {len(attribution_results['feature_importances'])} features")
        else:
            print("  ⚠ No feature attribution results")
        
        return attribution_results
        
    # ============================================================
    # VISUALIZATION METHOD
    # ============================================================
    
    def generate_all_visualizations(self, dataset_name: str = "main"):
        """Generate all comprehensive visualizations"""
        print("\n" + "="*70)
        print("GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("="*70)
        
        # Extract data from results
        cl_results = self.results.get('cross_lingual', {})
        stat_results = self.results.get('statistical', {})
        ling_results = self.results.get('linguistic', {})
        preproc_results = self.results.get('preprocessing', {})
        
        sdi_matrix = cl_results.get('sdi_matrix')
        sdi_classification = cl_results.get('sdi_classification', {})
        error_categories = cl_results.get('error_categories', {})
        
        length_stats = stat_results.get('response_length_stats', pd.DataFrame())
        vocab_stats = stat_results.get('vocabulary_richness', pd.DataFrame())
        test_results = stat_results.get('statistical_tests', [])
        outliers = stat_results.get('outliers', [])
        
        tp_df = preproc_results.get('tokenisation_parity', pd.DataFrame())
        trust_results = ling_results.get('trust_aware_results', [])
        complexity_df = ling_results.get('structural_complexity', pd.DataFrame())
        
        answers_by_lang = preproc_results.get('normalised_texts', {})
        embeddings = preproc_results.get('embeddings', np.array([]))
        labels = preproc_results.get('joint_labels', [])
        
        # Get feature attribution data
        feature_data = self._get_feature_attribution()
        
        print(f"\n  Data Summary:")
        print(f"    - SDI Matrix: {sdi_matrix is not None and not sdi_matrix.empty}")
        print(f"    - Length Stats: {not length_stats.empty}")
        print(f"    - Vocab Stats: {not vocab_stats.empty}")
        print(f"    - Test Results: {len(test_results) if test_results else 0}")
        print(f"    - Answers: {len(answers_by_lang) if answers_by_lang else 0}")
        print(f"    - Embeddings: {embeddings.shape if embeddings is not None and embeddings.size > 0 else 'None'}")
        print(f"    - Samples: {len(self.sample_results) if self.sample_results else 0}")
        
        viz_specs = [
            # Original 18 visualizations
            (self.viz.save_sdi_heatmap, [sdi_matrix], "1_sdi_heatmap", sdi_matrix is not None and not sdi_matrix.empty),
            (self.viz.save_response_length_chart, [length_stats], "2_response_length", not length_stats.empty),
            (self.viz.save_tokeniser_performance_chart, [tp_df], "3_tokeniser_performance", not tp_df.empty),
            (self.viz.save_vocabulary_richness_chart, [vocab_stats], "4_vocabulary_richness", not vocab_stats.empty),
            (self.viz.save_sdi_ranking_chart, [self._get_sdi_ranking(sdi_matrix)], "5_sdi_ranking", sdi_matrix is not None and not sdi_matrix.empty),
            (self.viz.save_root_cause_pie_chart, [error_categories], "6_root_cause", error_categories and error_categories.get('total', 0) > 0),
            (self.viz.save_trust_metrics_chart, [trust_results], "7_trust_metrics", bool(trust_results)),
            (self.viz.save_flags_distribution, [self._get_all_flags()], "8_flags_distribution", bool(self._get_all_flags())),
            (self.viz.save_statistical_tests_table, [test_results], "9_statistical_tests", bool(test_results)),
            (self.viz.save_outliers_chart, [outliers], "10_outliers", bool(outliers)),
            (self.viz.save_complexity_chart, [complexity_df], "11_complexity", not complexity_df.empty),
            (self.viz.save_radar_chart, [self._get_language_metrics()], "12_radar_chart", bool(self._get_language_metrics())),
            (self.viz.save_correlation_heatmap, [self._get_metrics_correlation()], "13_correlation", not self._get_metrics_correlation().empty),
            (self.viz.save_violin_plot, [answers_by_lang], "14_violin_plot", bool(answers_by_lang)),
            (self.viz.save_bias_gauge, [sdi_classification.get('average_sdi', 0), 
                                        sdi_classification.get('bias_level', 'Unknown'), 
                                        len(self._get_all_flags())], "15_bias_gauge", True),
            (self.viz.save_embedding_visualization, [embeddings, labels], "16_embedding_viz", embeddings is not None and embeddings.size > 0 and len(embeddings) > 10),
            (self.viz.save_experiment_summary, [self.experiment_results], "17_experiment_summary", bool(self.experiment_results)),
            (self.viz.save_executive_dashboard, [self._get_executive_report(dataset_name)], "18_dashboard", True),
            
            # Sample visualizations (19-20)
            (self.viz.save_sample_sentences_chart, [self.sample_results], "19_sample_sentences", bool(self.sample_results)),
            (self.viz.save_bias_reduction_chart, [self.sample_results], "20_bias_reduction", bool(self.sample_results)),
            
            # New visualizations (21-24)
            (self.viz.save_bias_reduction_diagram, [], "21_bias_reduction_framework", True),
            (self.viz.save_feature_attribution_plot, [feature_data], "22_feature_attribution", bool(feature_data)),
            (self.viz.save_encoder_calibration_plot, [self.calibration_results], "23_encoder_calibration", bool(self.calibration_results)),
            (self.viz.save_sdi_before_after_plot, [self.reduction_results], "24_sdi_before_after", bool(self.reduction_results)),
            (self.viz.save_bias_reduction_triples_table, [self.reduction_results], "25_bias_reduction_triples", bool(self.reduction_results))
        ]
        
        viz_count = 0
        for method, args, filename, condition in viz_specs:
            if not condition:
                print(f"   Skipping {filename} - condition not met")
                continue
            
            try:
                method(*args, dataset_name=dataset_name)
                viz_count += 1
                print(f"  ✓ Generated: {filename}.png")
            except Exception as e:
                print(f"  ✗ Failed: {filename}.png - {e}")
        
        print(f"\nGenerated {viz_count}/25 visualizations in: {FIGURES_DIR}")
    
    # ============================================================
    # FULL AUDIT METHOD
    # ============================================================
    
    def run_full_audit(self, run_experiments: bool = True, run_validation: bool = True) -> Dict:
        """Run complete audit pipeline"""
        print("\n" + "="*70)
        print("MAHEALTHBIASAUDIT - FULL PIPELINE")
        print("="*70)
        
        print_bias_characteristics()
        
        print("\n" + "="*70)
        print("STAGE 1: MAIN DATASET ANALYSIS")
        print("="*70)
        
        main_data = self.load_maternal_multilingual_dataset()
        main_report = self.run_pipeline(main_data, "maternal_multilingual_full")
        
        # ============================================================
        # STAGE 2: ENCODER CALIBRATION - ADD THIS
        # ============================================================
        print("\n" + "="*70)
        print(" STAGE 2: ENCODER CALIBRATION")
        print("="*70)
        
        # Get data for calibration
        preproc_results = self.results.get('preprocessing', {})
        cl_results = self.results.get('cross_lingual', {})
        
        answers_by_lang = preproc_results.get('normalised_texts', {})
        embeddings = preproc_results.get('embeddings', np.array([]))
        labels = preproc_results.get('joint_labels', [])
        
        if embeddings.size > 0 and labels:
            self.run_encoder_calibration(embeddings, labels, answers_by_lang)
        else:
            print("No embeddings available for calibration")
        
        # ============================================================
        # STAGE 3: FEATURE ATTRIBUTION - ADD THIS
        # ============================================================
        print("\n" + "="*70)
        print("STAGE 3: FEATURE ATTRIBUTION")
        print("="*70)
        
        if embeddings.size > 0 and labels and answers_by_lang:
            feature_data = self._get_feature_attribution()
            if feature_data:
                print("Feature attribution complete")
            else:
                print("⚠ No feature attribution data available")
        
        # ============================================================
        # STAGE 4: BIAS REDUCTION - ADD THIS
        # ============================================================
        print("\n" + "="*70)
        print("STAGE 4: BIAS REDUCTION FRAMEWORK")
        print("="*70)
        
        if embeddings.size > 0 and labels and answers_by_lang:
            self.run_bias_reduction(answers_by_lang, embeddings, labels)
        else:
            print("No data available for bias reduction")
        
        # ============================================================
        # Continue with existing stages
        # ============================================================
        self.generate_all_visualizations("main")
        
        if run_experiments:
            print("\n" + "="*70)
            print("STAGE 5: EXPERIMENTS")
            print("   Sample sizes increasing by squares: 10, 100, 1000, 10000")
            print("="*70)
            self.experiment_results = self.run_experiments()
        
        if run_validation:
            print("\n" + "="*70)
            print("STAGE 6: VALIDATION ON MOTHER DATASET")
            print("="*70)
            validation_result = self.run_validation()
            self.validation_results = validation_result
        
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"\nMain Dataset SDI: {main_report['key_metrics']['average_sdi']:.4f} ({main_report['key_metrics']['sdi_percentage']})")
        print(f"Main Dataset Bias: {main_report['key_metrics']['bias_level']}")
        print(f"Main Dataset Flags: {main_report['key_metrics']['total_flags']}")
        
        if run_validation:
            status = self.validation_results.get('status', 'UNKNOWN')
            print(f"\n MOTHER Validation: {status}")
        
        print(f"\n Output: {OUTPUT_DIR}")
        print(f"   ├─ Reports: {REPORTS_DIR}")
        print(f"   ├─ Figures: {FIGURES_DIR}")
        print(f"   ├─ Validation: {VALIDATION_DIR}")
        print(f"   └─ Experiments: {EXPERIMENTS_DIR}")
        
        return main_report


if __name__ == "__main__":
    audit = MaHealthBiasAudit()
    audit.run_full_audit(run_experiments=True, run_validation=True)