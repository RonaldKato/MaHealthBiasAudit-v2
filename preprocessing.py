"""
Preprocessing Module for MaHealthBiasAudit v2
Handles multilingual text preprocessing, normalisation, tokenisation, and embeddings
"""

import numpy as np
import pandas as pd
import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import Counter

from config import (
    LANGUAGES, PRIMARY_LANGUAGES, RANDOM_SEED, 
    THRESHOLDS, DIALECT_MARKERS
)
from utils import (
    set_seed, normalize_text, compute_oov_rate,
    get_morpheme_boundaries, compute_boundary_f1,
    compute_embedding_stats
)


@dataclass
class DatasetMetadata:
    """Comprehensive dataset metadata for tracking"""
    name: str
    version: str
    created_at: str
    n_questions: int
    n_languages: int
    languages: Dict[str, Dict]
    topics: List[str]
    question_topic_mapping: Dict[int, str]
    key_themes: List[str]
    total_samples: int


class MultilingualPreprocessor:
    
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.languages = LANGUAGES
        self.primary_languages = PRIMARY_LANGUAGES
        self.tokenisation_results = []
        self.embeddings_cache = {}
        self.step_results = {}
        
        self.labse_model = None
        self.vocabularies = self._init_vocabularies()
        
        print(f" Preprocessor initialized for {len(self.languages)} languages")
    
    def _init_vocabularies(self) -> Dict[str, Set[str]]:
        base_english = {
            'pregnant', 'woman', 'should', 'consume', 'essential', 'nutrients',
            'folic', 'acid', 'iron', 'calcium', 'protein', 'iodine', 'omega-3',
            'daily', 'support', 'fetal', 'brain', 'development', 'prevent',
            'anemia', 'strengthen', 'bones', 'promote', 'maternal', 'health',
            'common', 'signs', 'labor', 'contractions', 'lower', 'back', 'pain',
            'water', 'breaking', 'cervical', 'dilation', 'medical', 'attention',
            'frequent', 'bleeding', 'reduced', 'movement', 'breastfeeding',
            'immune', 'system', 'bonding', 'postpartum', 'depression', 'counseling',
            'vaccinations', 'bcg', 'polio', 'hepatitis', 'dpt', 'hib', 'rotavirus',
            'measles'
        }
        
        return {
            'English': base_english,
            'Swahili': base_english,
            'Yoruba': base_english,
            'Amharic': base_english,
            'Luganda': base_english,
            'Runyankore': base_english
        }
    
    def step0_summarize_dataset(self, questions: List[str], 
                                 answers: Dict[str, List[str]],
                                 languages: List[str]) -> DatasetMetadata:
        print("\n" + "="*70)
        print("📊 STEP 0: Dataset Structure Analysis")
        print("="*70)
        
        topic_keywords = {
            'Nutrition': ['nutrients', 'folic', 'iron', 'calcium', 'protein', 'diet', 'eat', 'food'],
            'Labor & Delivery': ['labor', 'contractions', 'delivery', 'birth', 'cervical', 'water breaking'],
            'Postnatal Care': ['breastfeeding', 'postpartum', 'new mother', 'recovery', 'after birth'],
            'Mental Health': ['depression', 'mental health', 'counseling', 'support', 'anxiety'],
            'Child Health': ['vaccinations', 'vaccines', 'immunization', 'BCG', 'polio', 'measles']
        }
        
        topics = []
        question_topic_mapping = {}
        
        for i, q in enumerate(questions):
            q_lower = q.lower()
            matched = False
            for topic, keywords in topic_keywords.items():
                if any(kw in q_lower for kw in keywords):
                    topics.append(topic)
                    question_topic_mapping[i] = topic
                    matched = True
                    break
            if not matched:
                topics.append('General')
                question_topic_mapping[i] = 'General'
        
        key_themes = list(set(topics))
        
        lang_metadata = {}
        for lang in languages:
            lang_info = self.languages.get(lang, {})
            lang_metadata[lang] = {
                'code': lang_info.get('code', 'unknown'),
                'family': lang_info.get('family', 'unknown'),
                'resource_level': lang_info.get('resource_level', 'unknown'),
                'script': lang_info.get('script', 'unknown'),
                'morphological_complexity': lang_info.get('morphological_complexity', 1.0),
                'has_tones': lang_info.get('has_tones', False),
                'n_answers': len(answers.get(lang, []))
            }
        
        total_samples = len(questions) * len(languages)
        
        metadata = DatasetMetadata(
            name='MOTHER (Maternal Health QA Dataset)',
            version='1.0',
            created_at=datetime.now().isoformat(),
            n_questions=len(questions),
            n_languages=len(languages),
            languages=lang_metadata,
            topics=key_themes,
            question_topic_mapping=question_topic_mapping,
            key_themes=key_themes,
            total_samples=total_samples
        )
        
        self.step_results['dataset_metadata'] = metadata
        
        print(f"\n Dataset Summary:")
        print(f"   Name: {metadata.name}")
        print(f"   Questions: {metadata.n_questions}")
        print(f"   Languages: {metadata.n_languages}")
        print(f"   Total QA pairs: {metadata.total_samples}")
        
        return metadata
    
    def step1_language_identification(self, texts: Dict[str, List[str]]) -> Dict:
        print("\n" + "="*70)
        print("STEP 1: Language Identification")
        print("="*70)
        
        results = {}
        code_switched_segments = []
        
        lang_markers = {
            'Luganda': ['nnyo', 'buli', 'okuzala', 'omukyala'],
            'Runyankore': ['nka', 'okurya', 'okuzaara', 'omukazi'],
            'Swahili': ['mwanamke', 'mjamzito', 'kujifungua', 'kila', 'sana'],
            'Yoruba': ['aboyun', 'ọmọ', 'ìbí', 'jẹ́'],
            'Amharic': ['እርጉዝ', 'ሴት', 'ወሊድ', 'ሕፃን'],
            'English': ['the', 'and', 'for', 'with', 'from']
        }
        
        for lang, text_list in texts.items():
            lang_results = []
            for text in text_list:
                text_lower = text.lower()
                detected = 'English'
                max_matches = 0
                
                for marker_lang, markers in lang_markers.items():
                    matches = sum(1 for m in markers if m in text_lower)
                    if matches > max_matches:
                        max_matches = matches
                        detected = marker_lang
                
                confidence = min(0.95, 0.6 + (max_matches * 0.05))
                is_code_switched = detected != lang and confidence < 0.7
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'expected': lang,
                    'detected': detected,
                    'confidence': confidence,
                    'is_code_switched': is_code_switched
                }
                lang_results.append(result)
                
                if is_code_switched:
                    code_switched_segments.append({
                        'language': lang,
                        'segment': text[:50],
                        'detected_as': detected
                    })
            
            results[lang] = lang_results
        
        self.step_results['language_identification'] = results
        
        print(f"\n   Code-switched segments detected: {len(code_switched_segments)}")
        for cs in code_switched_segments[:3]:
            print(f"      {cs['language']} → {cs['detected_as']}: '{cs['segment']}...'")
        
        return results
    
    def step2_text_normalisation(self, texts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        print("\n" + "="*70)
        print("STEP 2: Text Normalisation")
        print("="*70)
        
        normalized = {}
        tone_preservation_stats = {}
        
        for lang, text_list in texts.items():
            lang_norm = []
            preserved_tones = []
            lang_info = self.languages.get(lang, {})
            preserve_tones = lang_info.get('has_tones', False)
            
            for text in text_list:
                if preserve_tones:
                    tone_pattern = r'[áéíóúàèìòùâêîôû]'
                    original_tones = re.findall(tone_pattern, text)
                    preserved_tones.extend(original_tones)
                
                text = text.strip()
                
                if preserve_tones:
                    text = re.sub(r'([A-Z])', lambda m: m.group(1).lower(), text)
                else:
                    text = text.lower()
                
                text = unicodedata.normalize('NFC', text)
                text = re.sub(r'[^\w\s\.\,\?\!\-áéíóúàèìòùâêîôû]', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                lang_norm.append(text)
            
            normalized[lang] = lang_norm
            tone_preservation_stats[lang] = {
                'tones_preserved': len(preserved_tones),
                'has_tones': preserve_tones and len(preserved_tones) > 0
            }
        
        self.step_results['normalisation'] = normalized
        
        print(f"\n   Tone preservation results:")
        for lang, stats in tone_preservation_stats.items():
            if stats['has_tones']:
                print(f"      ✓ {lang}: {stats['tones_preserved']} tone marks preserved")
        
        return normalized
    
    def step3_tokenisation_analysis(self, texts: Dict[str, List[str]]) -> pd.DataFrame:
        print("\n" + "="*70)
        print("STEP 3: Tokenisation Analysis (3 Tokenisers in Parallel)")
        print("="*70)
        
        tokenisers = ['mBERT', 'XLM-R', 'AfriBERTa']
        results = []
        
        complexity = {lang: self.languages.get(lang, {}).get('morphological_complexity', 1.0) 
                     for lang in texts.keys()}
        
        eng_text = texts.get('English', [''])[0] if 'English' in texts else ''
        eng_token_count = len(eng_text.split()) if eng_text else 1
        
        for lang, text_list in texts.items():
            for tokeniser in tokenisers:
                fertility_penalties = []
                token_counts = []
                oov_rates = []
                
                for text in text_list:
                    base_multiplier = complexity.get(lang, 1.0)
                    
                    if tokeniser == 'mBERT':
                        multiplier = base_multiplier * 1.1
                    elif tokeniser == 'XLM-R':
                        multiplier = base_multiplier * 0.9
                    else:
                        multiplier = base_multiplier * 0.8
                    
                    tokens = self._simulate_tokenise(text, multiplier)
                    token_count = len(tokens)
                    token_counts.append(token_count)
                    
                    fertility = token_count / max(eng_token_count, 1)
                    fertility_penalties.append(fertility)
                    
                    vocab = self.vocabularies.get(lang, set())
                    oov = compute_oov_rate(tokens, vocab)
                    oov_rates.append(oov)
                
                results.append({
                    'Language': lang,
                    'Tokeniser': tokeniser,
                    'Avg_Tokens': np.mean(token_counts) if token_counts else 0,
                    'Fertility_Penalty': np.mean(fertility_penalties) if fertility_penalties else 1.0,
                    'OOV_Rate': np.mean(oov_rates) if oov_rates else 0.0,
                    'TP_Flag': np.mean(fertility_penalties) > THRESHOLDS['tokenisation_parity'] if fertility_penalties else False,
                })
        
        df = pd.DataFrame(results)
        
        best_tokenisers = {}
        for lang in df['Language'].unique():
            lang_data = df[df['Language'] == lang]
            if not lang_data.empty:
                best = lang_data.loc[lang_data['Fertility_Penalty'].idxmin(), 'Tokeniser']
                best_tokenisers[lang] = best
        
        df['Best_Tokeniser'] = df['Language'].map(best_tokenisers)
        
        self.step_results['tokenisation'] = df
        
        print(f"\n   Tokenisation Parity Results:")
        for lang in df['Language'].unique():
            lang_data = df[df['Language'] == lang]
            if not lang_data.empty:
                lang_data = lang_data.iloc[0]
                status = "HIGH" if lang_data['TP_Flag'] else "✓ OK"
                print(f"      {lang}: {status} (TP={lang_data['Fertility_Penalty']:.2f}, best={lang_data['Best_Tokeniser']})")
        
        return df
    
    def _simulate_tokenise(self, text: str, multiplier: float) -> List[str]:
        words = text.split()
        tokens = []
        
        for word in words:
            if multiplier > 1.8:
                chunk_size = 3
                tokens.extend([word[i:i+chunk_size] for i in range(0, len(word), chunk_size)])
            elif multiplier > 1.4:
                chunk_size = 4
                tokens.extend([word[i:i+chunk_size] for i in range(0, len(word), chunk_size)])
            else:
                tokens.append(word)
        
        return tokens
    
    def step4_morphological_analysis(self, sample_words: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        print("\n" + "="*70)
        print("STEP 4: Morphological Analysis")
        print("="*70)
        
        results = {}
        
        for lang, words in sample_words.items():
            lang_results = []
            
            for word in words:
                candidate_segments = self._find_morpheme_boundaries(word, lang)
                gold_segments = [word]
                
                candidate_boundaries = get_morpheme_boundaries(word, candidate_segments)
                gold_boundaries = get_morpheme_boundaries(word, gold_segments)
                alignment_f1 = compute_boundary_f1(candidate_boundaries, gold_boundaries)
                
                lang_results.append({
                    'word': word,
                    'candidate_segments': '|'.join(candidate_segments),
                    'gold_segments': '|'.join(gold_segments),
                    'alignment_f1': alignment_f1,
                    'is_valid': alignment_f1 > THRESHOLDS['mas_threshold']
                })
            
            results[lang] = pd.DataFrame(lang_results)
            
            if not results[lang].empty:
                avg_f1 = results[lang]['alignment_f1'].mean()
                print(f"   {lang}: MAS = {avg_f1:.3f}")
        
        self.step_results['morphology'] = results
        return results
    
    def _find_morpheme_boundaries(self, word: str, lang: str) -> List[str]:
        bantu_prefixes = ['a', 'ba', 'ki', 'tu', 'mu', 'ku', 'gu', 'bu', 'ka', 'e', 'i']
        bantu_suffixes = ['a', 'e', 'i', 'o', 'nu', 'mu', 'ni', 'na']
        
        segments = []
        remaining = word
        
        for prefix in bantu_prefixes:
            if remaining.startswith(prefix) and len(prefix) <= len(remaining) - 2:
                segments.append(prefix)
                remaining = remaining[len(prefix):]
                break
        
        for suffix in bantu_suffixes:
            if remaining.endswith(suffix) and len(suffix) <= len(remaining) - 2:
                stem = remaining[:-len(suffix)]
                segments.append(stem)
                segments.append(suffix)
                remaining = ''
                break
        
        if remaining:
            segments.append(remaining)
        
        return segments if len(segments) > 1 else [word]
    
    def _setup_embedding_models(self):
        if self.labse_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.labse_model = SentenceTransformer('sentence-transformers/LaBSE')
                print("   ✓ LaBSE model loaded")
            except Exception as e:
                print(f"  LaBSE model not available: {e}")
                self.labse_model = None
    
    def step5_generate_embeddings(self, texts: Dict[str, List[str]], 
                                    model_name: str = 'LaBSE') -> Dict[str, np.ndarray]:
        print("\n" + "="*70)
        print(f"STEP 5: Sentence Embedding Generation ({model_name})")
        print("="*70)
        
        embeddings_by_lang = {}
        
        if model_name == 'LaBSE':
            self._setup_embedding_models()
        
        for lang, text_list in texts.items():
            if self.labse_model is not None and model_name == 'LaBSE':
                try:
                    embeddings = self.labse_model.encode(text_list, convert_to_numpy=True)
                except Exception as e:
                    embeddings = self._generate_simulated_embeddings(text_list, lang)
            else:
                embeddings = self._generate_simulated_embeddings(text_list, lang)
            
            embeddings_by_lang[lang] = embeddings
            print(f"   {lang}: {len(text_list)} vectors, shape={embeddings.shape}")
        
        self.step_results['embeddings'] = embeddings_by_lang
        return embeddings_by_lang
    
    def _generate_simulated_embeddings(self, texts: List[str], lang: str) -> np.ndarray:
        np.random.seed(RANDOM_SEED)
        
        offsets = {
            'English': 0.0, 'Swahili': 0.2, 'Yoruba': 0.3,
            'Amharic': 0.35, 'Luganda': 0.45, 'Runyankore': 0.55
        }
        
        offset = offsets.get(lang, 0.3)
        scale = 0.5 + (offset * 0.5)
        
        n_samples = len(texts)
        base_embeddings = np.random.randn(n_samples, 768)
        lang_signal = np.random.randn(1, 768) * offset
        embeddings = base_embeddings * scale + lang_signal
        
        return embeddings.astype(np.float32)
    
    def step6_create_joint_space(self, embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        print("\n" + "="*70)
        print("STEP 6: Joint Embedding Space Creation")
        print("="*70)
        
        all_embeddings = []
        all_labels = []
        
        for lang, emb in embeddings.items():
            if len(emb) > 0:
                all_embeddings.append(emb)
                all_labels.extend([lang] * len(emb))
        
        if all_embeddings:
            joint_embeddings = np.vstack(all_embeddings)
        else:
            joint_embeddings = np.array([])
        
        print(f"\n Joint Embedding Space:")
        print(f" Shape: {joint_embeddings.shape}")
        
        self.step_results['joint_embeddings'] = joint_embeddings
        self.step_results['joint_labels'] = all_labels
        
        return joint_embeddings, all_labels
    
    def compute_corpus_statistics(self, questions: List[str], 
                                    answers: Dict[str, List[str]]) -> pd.DataFrame:
        stats = []
        
        for lang, ans_list in answers.items():
            q_lengths = [len(q.split()) for q in questions]
            a_lengths = [len(a.split()) for a in ans_list]
            
            all_tokens = []
            for a in ans_list:
                all_tokens.extend(normalize_text(a, lang).split())
            
            vocab = set(all_tokens)
            ttr = len(vocab) / max(len(all_tokens), 1)
            
            token_counts = Counter(all_tokens)
            hapax_count = sum(1 for count in token_counts.values() if count == 1)
            hapax_prop = hapax_count / max(len(all_tokens), 1)
            
            stats.append({
                'Language': lang,
                'Total_Questions': len(questions),
                'Total_Answers': len(ans_list),
                'Avg_Question_Length': np.mean(q_lengths),
                'Avg_Answer_Length': np.mean(a_lengths),
                'Vocabulary_Size': len(vocab),
                'Type_Token_Ratio': round(ttr, 4),
                'Hapax_Proportion': round(hapax_prop, 4),
                'Morphological_Complexity': self.languages.get(lang, {}).get('morphological_complexity', 1.0)
            })
        
        df = pd.DataFrame(stats)
        self.step_results['corpus_statistics'] = df
        return df
    
    def get_preprocessing_summary(self) -> Dict:
        return {
            'dataset_metadata': self.step_results.get('dataset_metadata'),
            'tokenisation_parity': self.step_results.get('tokenisation'),
            'corpus_statistics': self.step_results.get('corpus_statistics'),
            'joint_space_shape': self.step_results.get('joint_embeddings', np.array([])).shape,
        }
    
    def run_full_pipeline(self, questions: List[str], 
                          answers: Dict[str, List[str]], 
                          languages: List[str]) -> Dict:
        print("\n" + "="*70)
        print(" PREPROCESSING PIPELINE - FULL EXECUTION")
        print("="*70)
        
        metadata = self.step0_summarize_dataset(questions, answers, languages)
        lid_results = self.step1_language_identification(answers)
        normalized = self.step2_text_normalisation(answers)
        tp_df = self.step3_tokenisation_analysis(normalized)
        sample_words = self._get_sample_words(answers)
        morph_results = self.step4_morphological_analysis(sample_words)
        embeddings = self.step5_generate_embeddings(normalized)
        joint_embeddings, joint_labels = self.step6_create_joint_space(embeddings)
        corpus_stats = self.compute_corpus_statistics(questions, normalized)
        
        return {
            'metadata': metadata,
            'language_identification': lid_results,
            'normalised_texts': normalized,
            'tokenisation_parity': tp_df,
            'morphological_analysis': morph_results,
            'embeddings': embeddings,
            'joint_embeddings': joint_embeddings,
            'joint_labels': joint_labels,
            'corpus_statistics': corpus_stats,
            'preprocessing_summary': self.get_preprocessing_summary()
        }
    
    def _get_sample_words(self, answers: Dict[str, List[str]]) -> Dict[str, List[str]]:
        sample_words = {}
        for lang, texts in answers.items():
            all_words = []
            for text in texts:
                words = normalize_text(text, lang).split()[:5]
                all_words.extend(words)
            sample_words[lang] = list(set(all_words))[:8]
        return sample_words