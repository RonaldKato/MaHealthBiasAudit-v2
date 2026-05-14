"""
Stratum III: Model-Level Bias Audit
Includes: SLM fine-tuning, performance metrics, cross-lingual transfer analysis
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from config import MODEL_CONFIGS
import warnings
warnings.filterwarnings('ignore')

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

class QADataset(Dataset):
    """Dataset for QA fine-tuning"""
    def __init__(self, questions: List[str], answers: List[str], tokenizer, max_length: int = 128):
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # Tokenize input
        encoding = self.tokenizer(
            question,
            answer,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For language modeling
        }

class QAModel(nn.Module):
    """QA model wrapper for fine-tuning"""
    def __init__(self, model_name: str, num_labels: int = 2):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.qa_outputs = nn.Linear(self.model.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        return logits

class ModelBiasAuditor:
    """Stratum III: Model-level bias audit and evaluation"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.models = {
            'mBERT': 'bert-base-multilingual-cased',
            'XLM-R': 'xlm-roberta-base',
            'SERENGETI': 'Davlan/afro-xlmr-base'  # Proxy for SERENGETI
        }
        self.tokenizers = {}
        self.model_instances = {}
        self.performance_records: List[ModelPerformance] = []
        
        # Initialize models and tokenizers
        for model_name, model_path in self.models.items():
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
            self.model_instances[model_name] = QAModel(model_path).to(device)
    
    def prepare_data_splits(self, 
                            questions: Dict[str, List[str]],
                            answers: Dict[str, List[str]],
                            train_ratio: float = 0.8) -> Dict:
        """Prepare training and test splits for each language"""
        splits = {}
        
        for language in questions.keys():
            n_samples = len(questions[language])
            indices = np.random.permutation(n_samples)
            split_idx = int(n_samples * train_ratio)
            
            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
            
            splits[language] = {
                'train': {
                    'questions': [questions[language][i] for i in train_idx],
                    'answers': [answers[language][i] for i in train_idx]
                },
                'test': {
                    'questions': [questions[language][i] for i in test_idx],
                    'answers': [answers[language][i] for i in test_idx]
                }
            }
        
        return splits
    
    def fine_tune(self, 
                  model_name: str,
                  train_data: Dict[str, List[str]],
                  epochs: int = 3,
                  learning_rate: float = 2e-5,
                  batch_size: int = 8) -> QAModel:
        """Fine-tune model on training data"""
        model = self.model_instances[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Prepare dataset
        all_questions = []
        all_answers = []
        for lang in train_data.keys():
            all_questions.extend(train_data[lang]['questions'])
            all_answers.extend(train_data[lang]['answers'])
        
        dataset = QADataset(all_questions, all_answers, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                
                # Compute loss (simplified: cross-entropy on token predictions)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        model.eval()
        return model
    
    def evaluate(self,
                 model: QAModel,
                 tokenizer,
                 test_data: Dict[str, List[str]],
                 model_name: str,
                 training_condition: str) -> List[ModelPerformance]:
        """Evaluate model performance on test data"""
        performances = []
        
        for language, data in test_data.items():
            predictions = []
            ground_truths = []
            
            for question, answer in zip(data['questions'], data['answers']):
                # Predict answer span
                inputs = tokenizer(question, return_tensors='pt', truncation=True).to(self.device)
                
                with torch.no_grad():
                    logits = model(inputs['input_ids'], inputs['attention_mask'])
                
                # Get predicted span (simplified: take argmax)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_idx = torch.argmax(start_logits.squeeze()).item()
                end_idx = torch.argmax(end_logits.squeeze()).item()
                
                # Decode prediction
                input_ids = inputs['input_ids'][0]
                predicted_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx+1])
                predicted_answer = tokenizer.convert_tokens_to_string(predicted_tokens)
                
                predictions.append(predicted_answer)
                ground_truths.append(answer)
            
            # Compute metrics
            exact_matches = sum(1 for p, g in zip(predictions, ground_truths) 
                              if p.strip().lower() == g.strip().lower())
            exact_match = exact_matches / max(len(predictions), 1)
            
            # Token-level F1
            token_f1_scores = []
            for p, g in zip(predictions, ground_truths):
                p_tokens = set(p.lower().split())
                g_tokens = set(g.lower().split())
                if p_tokens or g_tokens:
                    intersection = len(p_tokens & g_tokens)
                    precision = intersection / max(len(p_tokens), 1)
                    recall = intersection / max(len(g_tokens), 1)
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                    else:
                        f1 = 0.0
                    token_f1_scores.append(f1)
            
            token_f1 = np.mean(token_f1_scores) if token_f1_scores else 0.0
            
            # Compute other metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                ground_truths, predictions, average='weighted', zero_division=0
            )
            
            # Character-level F2 for morphologically rich languages
            chr_f2 = self._compute_chr_f2(predictions, ground_truths)
            
            perf = ModelPerformance(
                model_name=model_name,
                language=language,
                training_condition=training_condition,
                exact_match=exact_match,
                token_f1=token_f1,
                precision=precision,
                recall=recall,
                chr_f2=chr_f2,
                perplexity=0.0  # Placeholder
            )
            performances.append(perf)
            self.performance_records.append(perf)
        
        return performances
    
    def _compute_chr_f2(self, predictions: List[str], ground_truths: List[str]) -> float:
        """Compute character-level F2 score"""
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            # Character n-gram overlap (n=2 for simplicity)
            pred_chars = set(pred.lower())
            truth_chars = set(truth.lower())
            
            if pred_chars or truth_chars:
                intersection = len(pred_chars & truth_chars)
                precision = intersection / max(len(pred_chars), 1)
                recall = intersection / max(len(truth_chars), 1)
                if precision + recall > 0:
                    # F2 gives more weight to recall
                    f2 = 5 * precision * recall / (4 * precision + recall) if precision > 0 else 0
                else:
                    f2 = 0.0
                scores.append(f2)
        
        return np.mean(scores) if scores else 0.0
    
    def compute_bias_metrics(self) -> pd.DataFrame:
        """Compute bias metrics including F1 disparity and transfer gain"""
        records = [vars(r) for r in self.performance_records]
        df = pd.DataFrame(records)
        
        # Compute delta_F1 (disparity from English baseline)
        eng_baseline = df[(df['training_condition'] == 'FT-EN') & (df['language'] == 'English')]
        
        if not eng_baseline.empty:
            baseline_f1 = eng_baseline['token_f1'].iloc[0]
            df['f1_disparity'] = baseline_f1 - df['token_f1']
            df['relative_disparity'] = df['f1_disparity'] / baseline_f1
        
        # Compute transfer gain for each language
        for lang in df['language'].unique():
            if lang != 'English':
                eng_trained = df[(df['training_condition'] == 'FT-EN') & (df['language'] == lang)]
                lang_trained = df[(df['training_condition'] == f'FT-{lang[:2].upper()}') & (df['language'] == lang)]
                
                if not eng_trained.empty and not lang_trained.empty:
                    gain = lang_trained['token_f1'].iloc[0] - eng_trained['token_f1'].iloc[0]
                    df.loc[(df['language'] == lang), 'transfer_gain'] = gain
        
        return df
    
    def run_full_experiment(self,
                            questions: Dict[str, List[str]],
                            answers: Dict[str, List[str]],
                            save_path: Optional[str] = None) -> pd.DataFrame:
        """Run the full 5x3 experiment matrix"""
        # Prepare data splits
        splits = self.prepare_data_splits(questions, answers)
        
        # Define training conditions
        training_conditions = [
            ('FT-EN', ['English']),
            ('FT-SW', ['Swahili']),
            ('FT-YO', ['Yoruba']),
            ('FT-AM', ['Amharic']),
            ('FT-MULTI', list(questions.keys()))
        ]
        
        all_results = []
        
        for model_name in self.models.keys():
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print('='*50)
            
            for condition, train_langs in training_conditions:
                print(f"\nCondition: {condition}")
                
                # Prepare training data
                train_data = {}
                for lang in train_langs:
                    if lang in splits:
                        train_data[lang] = splits[lang]['train']
                
                # Fine-tune model
                model = self.fine_tune(model_name, train_data, epochs=2)  # Reduced epochs for demo
                
                # Evaluate on all languages
                test_data = {lang: splits[lang]['test'] for lang in splits.keys()}
                results = self.evaluate(model, self.tokenizers[model_name], 
                                       test_data, model_name, condition)
                
                for r in results:
                    all_results.append(vars(r))
        
        results_df = pd.DataFrame(all_results)
        
        if save_path:
            results_df.to_csv(save_path, index=False)
        
        return results_df


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_questions = {
        'English': [
            "What are essential nutrients for pregnancy?",
            "What are signs of labor?",
            "Benefits of breastfeeding?",
            "Postpartum depression support?",
            "Childhood vaccinations?"
        ],
        'Swahili': [
            "Virutubisho muhimu kwa ujauzito?",
            "Dalili za uchungu wa kujifungua?",
            "Faida za kunyonyesha?",
            "Usaidizi wa mfadhaiko baada ya kujifungua?",
            "Chanjo za utoto?"
        ],
        'Yoruba': [
            "Awọn eroja pataki fun oyun?",
            "Awọn ami ibi?",
            "Awọn anfani ti mimu omu?",
            "Atilẹyin ibanujẹ lẹyin ibi?",
            "Awọn ajesara ọmọde?"
        ],
        'Amharic': [
            "ለእርግዝና አስፈላጊ ንጥረ ነገሮች?",
            "የወሊድ ምልክቶች?",
            "የጡት ማጥባት ጥቅሞች?",
            "የወሊድ ኋላ ድብርት ድጋፍ?",
            "የልጅነት ክትባቶች?"
        ]
    }
    
    sample_answers = {
        'English': [
            "Folic acid, iron, calcium, protein",
            "Regular contractions, water breaking",
            "Strengthens immunity, bonding",
            "Counseling, support groups",
            "BCG, Polio, Measles"
        ],
        'Swahili': [
            "Asidi ya foliki, chuma, kalsiamu, protini",
            "Mikazo ya mara kwa mara, kupasuka kwa maji",
            "Huimarisha kinga, uhusiano",
            "Ushauri nasaha, vikundi vya msaada",
            "BCG, Polio, Surua"
        ],
        'Yoruba': [
            "Folic acid, iron, kalisiumu, amuaradagba",
            "Ìrora tí ń bọ leralera, omi ìyá tí ń já",
            "Ó fún ní agbára ààbò, ìbáṣepọ̀",
            "Ìmọ̀ràn, ẹgbẹ́ atilẹyin",
            "BCG, Polio, Measles"
        ],
        'Amharic': [
            "ፎሊክ አሲድ፣ ብረት፣ ካልሲየም፣ ፕሮቲን",
            "ቀጣይ ምጥ፣ የውሃ ፍሰት",
            "ኢምዩን ሲስተም ያጠናክራል፣ ትስስር",
            "ምክር አገልግሎት፣ የድጋፍ ቡድኖች",
            "BCG፣ ፖሊዮ፣ ኩፍኝ"
        ]
    }
    
    # Initialize auditor
    auditor = ModelBiasAuditor()
    
    # Run experiment (reduced size for demo)
    results = auditor.run_full_experiment(sample_questions, sample_answers)
    print("\nModel Performance Results:")
    print(results)
    
    # Compute bias metrics
    bias_metrics = auditor.compute_bias_metrics()
    print("\nBias Metrics:")
    print(bias_metrics)