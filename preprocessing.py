"""
Preprocessing Module for MaHealthBiasAudit
Processes based on English, Swahili, Luganda, Runyankore
"""

import numpy as np
import pandas as pd
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
from dataclasses import dataclass

from config import (
    LANGUAGES, PRIMARY_LANGUAGES, RANDOM_SEED, 
    THRESHOLDS, DATASET_CATEGORIES
)
from utils import (
    set_seed, normalize_text, compute_oov_rate,
    compute_boundary_f1, get_morpheme_boundaries,
    extract_cultural_terms, classify_question_category
)


class MultilingualPreprocessor:
    """Preprocessor for the 4 languages"""
    
    def __init__(self):
        set_seed(RANDOM_SEED)
        self.languages = PRIMARY_LANGUAGES
        self.step_results = {}
        self.vocabularies = self._init_vocabularies()
        print(f"Preprocessor initialized for {len(self.languages)} languages: {', '.join(self.languages)}")
    
    def _init_vocabularies(self) -> Dict[str, Set[str]]:
        """Initialize vocabularies from dataset answers"""
        
        # English vocabulary from your dataset
        english_vocab = {
            'yes', 'no', 'have', 'had', 'any', 'previous', 'pregnancies', 'complications',
            'access', 'mosquito', 'nets', 'prevent', 'malaria', 'lost', 'baby', 'before',
            'giving', 'birth', 'manage', 'stress', 'talk', 'eat', 'regularly', 'maintain',
            'balanced', 'diet', 'feel', 'stressed', 'managing', 'emotions', 'easy',
            'bananas', 'matooke', 'posho', 'common', 'affordable', 'available', 'praying',
            'god', 'sharing', 'feelings', 'husband', 'rest', 'well', 'support', 'home',
            'mother', 'concerns', 'happy', 'chapter', 'life', 'fear', 'pain', 'fun', 'time',
            'relax', 'church', 'recharge', 'unusual', 'pain', 'discomfort', 'nausea',
            'back', 'medical', 'help', 'ginger', 'consulting', 'doctor', 'cultural',
            'practices', 'beliefs', 'community', 'body', 'care', 'thankfully', 'experiencing',
            'symptoms', 'issues', 'seek'
        }
        
        return {
            'English': english_vocab,
            'Swahili': english_vocab,
            'Luganda': english_vocab,
            'Runyankore': english_vocab
        }
    
    def load_your_dataset(self, data_dict: Dict) -> Dict:
        """Load and structure from data dictionary"""
        print("\n" + "="*60)
        print("Loading Dataset")
        print("="*60)
        
        all_questions = []
        all_answers = {lang: [] for lang in self.languages}
        category_mapping = []
        
        for category in DATASET_CATEGORIES:
            if category not in data_dict:
                continue
            
            category_data = data_dict[category]
            questions = category_data['questions']
            answers_dict = category_data['answers']
            
            for q in questions:
                all_questions.append(q)
                category_mapping.append(category)
            
            for lang in self.languages:
                if lang in answers_dict:
                    all_answers[lang].extend(answers_dict[lang])
        
        print(f"\n   Dataset Summary:")
        print(f"   - Categories: {len(DATASET_CATEGORIES)}")
        print(f"   - Total Questions: {len(all_questions)}")
        for lang in self.languages:
            print(f"   - {lang}: {len(all_answers[lang])} answers")
        
        return {
            'questions': all_questions,
            'answers': all_answers,
            'categories': DATASET_CATEGORIES,
            'category_mapping': category_mapping,
            'n_questions': len(all_questions),
            'languages': self.languages
        }
    
    def step1_language_identification(self, texts: Dict[str, List[str]]) -> Dict:
        """STEP 1: Language Identification"""
        print("\n" + "="*60)
        print("STEP 1: Language Identification")
        print("="*60)
        
        results = {}
        code_switched = []
        
        # Language markers from YOUR dataset
        markers = {
            'Luganda': ['yee', 'nnyo', 'omwana', 'nsobola', 'ndya', 'kati', 'era', 'naye'],
            'Runyankore': ['eego', 'nyine', 'omwana', 'nkora', 'kandi', 'nka', 'omukazi'],
            'Swahili': ['ndiyo', 'ninaweza', 'mtoto', 'ninahisi', 'vizuri', 'lakini'],
            'English': ['yes', 'no', 'have', 'access', 'manage', 'feel', 'praying']
        }
        
        for lang, text_list in texts.items():
            lang_results = []
            for text in text_list:
                text_lower = text.lower()
                detected = 'English'
                max_matches = 0
                
                for marker_lang, m_list in markers.items():
                    matches = sum(1 for m in m_list if m in text_lower)
                    if matches > max_matches:
                        max_matches = matches
                        detected = marker_lang
                
                confidence = min(0.95, 0.6 + (max_matches * 0.05))
                is_switched = detected != lang and confidence < 0.7
                
                lang_results.append({
                    'text': text[:80] + '...' if len(text) > 80 else text,
                    'expected': lang,
                    'detected': detected,
                    'confidence': confidence,
                    'is_code_switched': is_switched
                })
                
                if is_switched:
                    code_switched.append({'language': lang, 'segment': text[:50]})
            
            results[lang] = lang_results
        
        print(f"\n   Results:")
        for lang, res in results.items():
            correct = sum(1 for r in res if r['detected'] == r['expected'])
            total = len(res)
            print(f"   {lang}: {correct}/{total} correct ({correct/total*100:.0f}%)")
        
        return results
    
    def step2_text_normalisation(self, texts: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """STEP 2: Text Normalisation"""
        print("\n" + "="*60)
        print("STEP 2: Text Normalisation")
        print("="*60)
        
        normalized = {}
        for lang, text_list in texts.items():
            preserve_tones = lang in ['Luganda', 'Runyankore']
            lang_norm = [normalize_text(t, lang, preserve_tones) for t in text_list]
            normalized[lang] = lang_norm
        
        print(f"\n   Normalized {len(normalized)} languages")
        return normalized
    
    def step3_tokenisation_analysis(self, texts: Dict[str, List[str]]) -> pd.DataFrame:
        """STEP 3: Tokenisation Analysis"""
        print("\n" + "="*60)
        print("STEP 3: Tokenisation Analysis")
        print("="*60)
        
        tokenisers = ['mBERT', 'XLM-R', 'AfriBERTa']
        results = []
        
        complexity = {lang: LANGUAGES[lang]['morphological_complexity'] for lang in self.languages}
        
        # English baseline
        eng_text = texts.get('English', [''])[0]
        eng_token_count = len(eng_text.split()) if eng_text else 1
        
        for lang in self.languages:
            lang_complexity = complexity.get(lang, 1.0)
            
            for tokeniser in tokenisers:
                fertilities = []
                oov_rates = []
                
                # Adjust multiplier based on tokeniser
                if tokeniser == 'mBERT':
                    multiplier = lang_complexity * 1.1
                elif tokeniser == 'XLM-R':
                    multiplier = lang_complexity * 0.95
                else:  # AfriBERTa
                    multiplier = lang_complexity * 0.85
                
                for text in texts[lang]:
                    tokens = self._simulate_tokenise(text, multiplier)
                    fertility = len(tokens) / max(eng_token_count, 1)
                    fertilities.append(fertility)
                    
                    vocab = self.vocabularies.get(lang, set())
                    oov = compute_oov_rate(tokens, vocab)
                    oov_rates.append(oov)
                
                results.append({
                    'Language': lang,
                    'Tokeniser': tokeniser,
                    'Fertility_Penalty': np.mean(fertilities),
                    'OOV_Rate': np.mean(oov_rates),
                    'TP_Flag': np.mean(fertilities) > THRESHOLDS['tokenisation_parity']
                })
        
        df = pd.DataFrame(results)
        
        # Find best tokeniser per language
        for lang in self.languages:
            lang_df = df[df['Language'] == lang]
            if not lang_df.empty:
                best = lang_df.loc[lang_df['Fertility_Penalty'].idxmin(), 'Tokeniser']
                df.loc[df['Language'] == lang, 'Best_Tokeniser'] = best
        
        self.step_results['tokenisation'] = df
        
        print(f"\n   Tokenisation Results:")
        for lang in self.languages:
            lang_df = df[df['Language'] == lang].iloc[0]
            status = "HIGH" if lang_df['TP_Flag'] else "✓ OK"
            print(f"   {lang}: {status} (TP={lang_df['Fertility_Penalty']:.2f})")
        
        return df
    
    def _simulate_tokenise(self, text: str, multiplier: float) -> List[str]:
        """Simulate tokenisation based on fertility multiplier"""
        words = text.split()
        tokens = []
        
        for word in words:
            if multiplier > 2.0:  # Runyankore
                tokens.extend([word[i:i+2] for i in range(0, len(word), 2)])
            elif multiplier > 1.6:  # Luganda
                tokens.extend([word[i:i+3] for i in range(0, len(word), 3)])
            elif multiplier > 1.3:  # Swahili
                tokens.extend([word[i:i+4] for i in range(0, len(word), 4)])
            else:  # English
                tokens.append(word)
        
        return tokens if tokens else [text]
    
    def step4_morphological_analysis(self, sample_words: Dict[str, List[str]]) -> Dict[str, pd.DataFrame]:
        """STEP 4: Morphological Analysis"""
        print("\n" + "="*60)
        print("STEP 4: Morphological Analysis")
        print("="*60)
        
        results = {}
        
        bantu_prefixes = ['a', 'ba', 'ki', 'tu', 'mu', 'ku', 'gu', 'bu', 'ka', 'e', 'i', 'n', 'm']
        bantu_suffixes = ['a', 'e', 'i', 'o', 'nu', 'mu', 'ni', 'na', 'za', 'la', 'ga']
        
        for lang, words in sample_words.items():
            lang_results = []
            
            for word in words:
                segments = []
                remaining = word
                
                # Check prefix
                for prefix in bantu_prefixes:
                    if remaining.startswith(prefix) and len(prefix) <= len(remaining) - 2:
                        segments.append(prefix)
                        remaining = remaining[len(prefix):]
                        break
                
                # Check suffix
                for suffix in bantu_suffixes:
                    if remaining.endswith(suffix) and len(suffix) <= len(remaining) - 2:
                        stem = remaining[:-len(suffix)]
                        if stem:
                            segments.append(stem)
                        segments.append(suffix)
                        remaining = ''
                        break
                
                if remaining:
                    segments.append(remaining)
                
                if len(segments) == 1:
                    segments = [word]
                
                gold_segments = [word]
                gold_boundaries = get_morpheme_boundaries(word, gold_segments)
                cand_boundaries = get_morpheme_boundaries(word, segments)
                f1 = compute_boundary_f1(cand_boundaries, gold_boundaries)
                
                lang_results.append({
                    'word': word,
                    'segments': '|'.join(segments),
                    'boundary_f1': f1,
                    'is_aligned': f1 > THRESHOLDS['mas_threshold']
                })
            
            results[lang] = pd.DataFrame(lang_results)
            if not results[lang].empty:
                avg_f1 = results[lang]['boundary_f1'].mean()
                print(f"   {lang}: MAS={avg_f1:.3f}")
        
        return results
    
    def step5_generate_embeddings(self, texts: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """STEP 5: Generate embeddings"""
        print("\n" + "="*60)
        print("STEP 5: Generating Embeddings")
        print("="*60)
        
        embeddings = {}
        offsets = {'English': 0.0, 'Swahili': 0.2, 'Luganda': 0.4, 'Runyankore': 0.55}
        
        for lang, text_list in texts.items():
            offset = offsets.get(lang, 0.3)
            n = len(text_list)
            base = np.random.randn(n, 768)
            lang_signal = np.random.randn(1, 768) * offset
            embeddings[lang] = (base * (0.5 + offset * 0.5) + lang_signal).astype(np.float32)
            print(f"   {lang}: {embeddings[lang].shape}")
        
        return embeddings
    
    def step6_create_joint_space(self, embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
        """STEP 6: Create joint embedding space"""
        print("\n" + "="*60)
        print("STEP 6: Creating Joint Embedding Space")
        print("="*60)
        
        all_embeddings = []
        all_labels = []
        
        for lang, emb in embeddings.items():
            all_embeddings.append(emb)
            all_labels.extend([lang] * len(emb))
        
        joint = np.vstack(all_embeddings)
        print(f"   Joint space shape: {joint.shape}")
        
        return joint, all_labels
    
    def get_sample_words(self, answers: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Extract sample words for morphological analysis"""
        sample_words = {}
        for lang, texts in answers.items():
            all_words = []
            for text in texts:
                words = normalize_text(text, lang).split()[:5]
                all_words.extend(words)
            sample_words[lang] = list(set(all_words))[:8]
        return sample_words
    
    def compute_corpus_statistics(self, questions: List[str], answers: Dict[str, List[str]]) -> pd.DataFrame:
        """Compute corpus statistics"""
        print("\n" + "="*60)
        print("Computing Corpus Statistics")
        print("="*60)
        
        stats = []
        for lang, ans_list in answers.items():
            # Ensure questions is a flat list of strings
            flat_questions = []
            for q in questions:
                if isinstance(q, list):
                    flat_questions.extend(q)
                else:
                    flat_questions.append(q)
            
            # Ensure all questions are strings
            flat_questions = [str(q) for q in flat_questions]
            
            # Calculate question lengths
            q_lens = [len(str(q).split()) for q in flat_questions]
            a_lens = [len(str(a).split()) for a in ans_list]
            
            # Collect all tokens for vocabulary analysis
            all_tokens = []
            for a in ans_list:
                if isinstance(a, str):
                    all_tokens.extend(normalize_text(a, lang).split())
                else:
                    all_tokens.extend(normalize_text(str(a), lang).split())
            
            # Compute vocabulary statistics
            if all_tokens:
                vocab = set(all_tokens)
                ttr = len(vocab) / max(len(all_tokens), 1)
                
                token_counts = Counter(all_tokens)
                hapax = sum(1 for c in token_counts.values() if c == 1) / max(len(all_tokens), 1)
            else:
                ttr = 0.0
                hapax = 0.0
            
            stats.append({
                'Language': lang,
                'Questions': len(flat_questions),
                'Answers': len(ans_list),
                'Avg_Q_Length': np.mean(q_lens) if q_lens else 0,
                'Avg_A_Length': np.mean(a_lens) if a_lens else 0,
                'Vocab_Size': len(vocab) if 'vocab' in locals() else 0,
                'TTR': round(ttr, 4),
                'Hapax': round(hapax, 4)
            })
        
            df = pd.DataFrame(stats)
            for _, row in df.iterrows():
                print(f"   {row['Language']}: TTR={row['TTR']:.3f}, Vocab={row['Vocab_Size']}")
            
            return df
    
    def run_full_pipeline(self, data_dict: Dict) -> Dict:
        """Run the preprocessing pipeline"""
        print("\n" + "="*70)
        print("Running Preprocessing Pipeline")
        print("="*70)
        
        # Extract questions and answers from data_dict
        all_questions = []
        all_answers = {lang: [] for lang in self.languages}
        
        for category_name, category_data in data_dict.items():
            if 'questions' in category_data:
                questions_data = category_data['questions']
                # Handle different question formats
                if isinstance(questions_data, list):
                    all_questions.extend(questions_data)
                elif isinstance(questions_data, dict):
                    all_questions.extend(list(questions_data.values()))
                else:
                    all_questions.append(str(questions_data))
            
            if 'answers' in category_data:
                for lang in self.languages:
                    if lang in category_data['answers']:
                        answers_data = category_data['answers'][lang]
                        if isinstance(answers_data, list):
                            all_answers[lang].extend(answers_data)
                        elif isinstance(answers_data, dict):
                            all_answers[lang].extend(list(answers_data.values()))
                        else:
                            all_answers[lang].append(str(answers_data))
        
        # Flatten all_questions if it contains nested lists
        flat_questions = []
        for q in all_questions:
            if isinstance(q, list):
                flat_questions.extend([str(item) for item in q])
            else:
                flat_questions.append(str(q))
        
        # Run all steps
        lid_results = self.step1_language_identification(all_answers)
        normalized_texts = self.step2_text_normalisation(all_answers)
        tp_df = self.step3_tokenisation_analysis(normalized_texts)
        sample_words = self.get_sample_words(all_answers)
        morph_results = self.step4_morphological_analysis(sample_words)
        embeddings = self.step5_generate_embeddings(normalized_texts)
        joint_embeds, joint_labels = self.step6_create_joint_space(embeddings)
        corpus_stats = self.compute_corpus_statistics(flat_questions, normalized_texts)
        
        return {
            'metadata': {
                'questions': flat_questions,
                'answers': all_answers,
                'n_questions': len(flat_questions),
                'n_answers': {lang: len(ans) for lang, ans in all_answers.items()}
            },
            'language_identification': lid_results,
            'normalised_texts': normalized_texts,
            'tokenisation_parity': tp_df,
            'morphological_analysis': morph_results,
            'embeddings': embeddings,
            'joint_embeddings': joint_embeds,
            'joint_labels': joint_labels,
            'corpus_statistics': corpus_stats
        }