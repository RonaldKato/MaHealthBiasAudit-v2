#!/usr/bin/env python3
"""
MaHealthBiasAudit v2 - Complete Bias Detection Pipeline
Main entry point for the entire bias audit system

This pipeline executes:
1. Preprocessing 
2. Stratum I: Statistical Bias Audit
3. Stratum II: Linguistic Bias Audit
4. Stratum III: Model Bias Audit
5. Cross-Lingual Evaluation
6. Bias Tracking and Visualization
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Set environment variables before imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Import all modules
from config import (
    PRIMARY_LANGUAGES, LANGUAGES, THRESHOLDS, 
    OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR,
    RANDOM_SEED, EXPERIMENT_NAME, EXPERIMENT_VERSION
)
from utils import set_seed, classify_maternal_topic, save_report
from preprocessing import MultilingualPreprocessor
from stratum_i_statistical import StatisticalBiasAuditor
from stratum_ii_linguistic import LinguisticBiasAuditor
from stratum_iii_model import ModelBiasAuditor
from cross_lingual_evaluation import CrossLingualEvaluator
from bias_tracker import BiasTracker
from visualization_dashboard import BiasVisualizationDashboard


class MaHealthBiasAuditPipeline:
    """
    Complete pipeline for Multilingual Maternal Health QA Bias Audit
    Integrates all 4 strata and produces comprehensive bias reports
    """
    
    def __init__(self, save_visuals: bool = True, show_visuals: bool = False):
        """
        Initialize the complete pipeline
        
        Args:
            save_visuals: Whether to save visualization figures to disk
            show_visuals: Whether to display figures interactively
        """
        set_seed(RANDOM_SEED)
        self.save_visuals = save_visuals
        self.show_visuals = show_visuals
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Initialize all components
        print("\n" + "="*70)
        print(f" {EXPERIMENT_NAME} v{EXPERIMENT_VERSION}")
        print("="*70)
        
        self.preprocessor = MultilingualPreprocessor()
        self.stat_auditor = StatisticalBiasAuditor()
        self.ling_auditor = LinguisticBiasAuditor()
        self.model_auditor = ModelBiasAuditor()
        self.cross_lingual = CrossLingualEvaluator()
        self.bias_tracker = BiasTracker(save_path=f"{REPORTS_DIR}/bias_history.json")
        self.viz_dashboard = BiasVisualizationDashboard(
            save_figures=save_visuals,
            output_dir=FIGURES_DIR,
            show_display=show_visuals
        )
        
        # Storage for results
        self.results = {}
        
        print("\n All components initialized successfully")
        print(f"   Visualizations will be {'saved to ' + FIGURES_DIR if save_visuals else 'NOT saved'}")
    
    def load_data(self) -> Dict:
        """
        Load the maternal health QA dataset
        Based on the provided data in the attachment
        """
        print("\n" + "="*60)
        print(" Loading Dataset")
        print("="*60)
        
        # Core maternal health questions
        maternal_health_questions = [
            "What are the essential nutrients a pregnant woman should consume daily?",
            "What are the common signs of labor, and when should a pregnant woman seek medical attention?",
            "What are the benefits of breastfeeding for both the mother and the baby?",
            "How can a new mother cope with postpartum depression, and what support systems are available?",
            "What are the recommended vaccinations for a child from birth to one year of age, and why are they important?"
        ]
        
        # Answers in each language (from attachment)
        answers = {
            'English': [
                "A pregnant woman should consume folic acid, iron, calcium, protein, iodine, and omega-3 fatty acids daily. These nutrients support fetal brain development, prevent anemia, strengthen bones, and promote overall maternal health.",
                "Common signs include regular contractions, lower back pain, water breaking, and cervical dilation. Medical attention should be sought when contractions are frequent (every 5 minutes), if there is bleeding, or reduced fetal movement.",
                "Breastfeeding strengthens the baby's immune system, provides optimal nutrition, and promotes bonding. For mothers, it reduces postpartum bleeding, lowers the risk of certain cancers, and aids recovery.",
                "New mothers can cope through counseling, peer support groups, family support, and professional care. Early screening and mental health services are essential.",
                "Key vaccines include BCG, Polio, Hepatitis B, DPT, Hib, Pneumococcal, Rotavirus, and Measles. They protect against life-threatening infections and support child survival."
            ],
            'Swahili': [
                "Mwanamke mjamzito anapaswa kula virutubisho muhimu kama asidi ya foliki, chuma, kalsiamu, protini, iodini, na mafuta ya omega-3 kila siku. Virutubisho hivi husaidia ukuaji wa ubongo wa mtoto, kuzuia upungufu wa damu, na kuimarisha afya ya mama.",
                "Dalili za uchungu wa kujifungua ni pamoja na mikazo ya mara kwa mara, maumivu ya mgongo, kupasuka kwa maji ya uzazi, na kufunguka kwa mlango wa kizazi. Huduma ya afya inahitajika ikiwa mikazo ni ya mara kwa mara au kuna damu au mtoto hasogei.",
                "Kunyonyesha huimarisha kinga ya mtoto, humpa lishe bora, na huongeza uhusiano kati ya mama na mtoto. Kwa mama, hupunguza damu baada ya kujifungua na hatari ya saratani.",
                "Mama anaweza kukabiliana na msongo wa mawazo baada ya kujifungua kwa ushauri nasaha, vikundi vya msaada, familia, na huduma za afya ya akili. Uchunguzi wa mapema ni muhimu.",
                "Chanjo muhimu ni BCG, Polio, Hepatitis B, DPT, Hib, Pneumococcal, Rotavirus, na Surua. Huzuia magonjwa hatari na kulinda maisha ya mtoto."
            ],
            'Yoruba': [
                "Aboyun yẹ kí ó máa jẹ́ àwọn eroja pataki bíi folic acid, irin (iron), kalisiumu, amuaradagba (protein), iodini, àti omega-3 lojoojúmọ́. Wọ́n ń ran ọmọ lọ́wọ́ láti dagba dáadáa, wọ́n sì ń dáàbò bo ìlera ìyá.",
                "Àwọn àmi ìbí ni ìrora tí ń bọ leralera, ìrora ẹ̀yìn, omi ìyá tí ń já, àti ìṣíṣi ilé-ọmọ. Obìnrin yẹ kí ó lọ sí iléewosan tí ìrora bá ń wá lojoojúmọ́, tàbí tí ẹ̀jẹ̀ bá ń já tàbí ọmọ kò ń rìn.",
                "Mímu ọmu ń fún ọmọ ní agbára ààbò ara, ń pèsè oúnjẹ tó péye, ó sì ń mú ìbáṣepọ̀ pọ̀ sí i. Fún ìyá, ó dín ẹ̀jẹ̀ kù lẹ́yìn ìbí, ó sì dín ewu àrùn kan kù.",
                "Ìyá tuntun lè koju ìbànújẹ́ lẹ́yìn ìbí pẹ̀lú ìmọ̀ràn, ẹgbẹ́ atilẹyin, ìdílé, àti ìtọju ọjọ́gbọn. Ìwádìí kíákíá àti ìtìlẹ́yìn ṣe pàtàkì.",
                "Àwọn àbẹ́rẹ́ ajẹsára pàtàkì ni BCG, Polio, Hepatitis B, DPT, Hib, Pneumococcal, Rotavirus, àti Measles. Wọ́n ń dáàbò bo ọmọ kúrò nínú àrùn tó lewu."
            ],
            'Amharic': [
                "እርጉዝ ሴት በየቀኑ ፎሊክ አሲድ፣ ብረት፣ ካልሲየም፣ ፕሮቲን፣ አዮዲን እና ኦሜጋ-3 እንዲያገኝ ይገባል። እነዚህ ንጥረ ምግቦች የሕፃኑን እድገት ይደግፋሉ እና የእናቱን ጤና ያሻሽላሉ.",
                "የወሊድ ምልክቶች የሚያካትቱት ቀጣይ ምጥ፣ የጀርባ ህመም፣ የውሃ ፍሰት እና የማህፀን መክፈት ናቸው። ምጥ በተደጋጋሚ ሲመጣ ወይም ደም ሲፈስ ወይም ህፃኑ ካልንቀሳቀሰ ሕክምና ያስፈልጋል።",
                "ጡት ማጥባት የሕፃኑን ኢምዩን ስርዓት ያጠናክራል፣ ሙሉ ምግብ ይሰጣል እና እናትና ልጅ መቀራረብን ያበረታታል። ለእናቱም የወሊድ ኋላ ደም መፍሰስን ይቀንሳል።",
                "አዲስ እናቶች የወሊድ ኋላ ድብርትን በምክር አገልግሎት፣ በድጋፍ ቡድኖች፣ በቤተሰብ እና በሙያዊ እገዛ ሊቋቋሙ ይችላሉ።",
                "አስፈላጊ ክትባቶች የሚያካትቱት BCG፣ ፖሊዮ፣ ሄፓታይቲስ B፣ DPT፣ Hib፣ Pneumococcal፣ Rotavirus እና ኩፍኝ ናቸው። እነዚህ ህፃናትን ከከባድ በሽታዎች ይጠብቃሉ።"
            ]
        }
        
        # Structure data by language
        questions_by_lang = {lang: maternal_health_questions for lang in PRIMARY_LANGUAGES}
        answers_by_lang = {lang: answers.get(lang, answers['English']) for lang in PRIMARY_LANGUAGES}
        
        # Add topic classification for each question
        topics = [classify_maternal_topic(q) for q in maternal_health_questions]
        
        print(f"\n Data loaded:")
        print(f"   - {len(maternal_health_questions)} maternal health questions")
        print(f"   - {len(PRIMARY_LANGUAGES)} languages: {', '.join(PRIMARY_LANGUAGES)}")
        print(f"   - Topics: {', '.join(set(topics))}")
        
        return {
            'questions': questions_by_lang,
            'answers': answers_by_lang,
            'n_questions': len(maternal_health_questions),
            'languages': PRIMARY_LANGUAGES,
            'topics': topics
        }
    
    def run_preprocessing(self, data: Dict) -> Dict:
        """Run preprocessing pipeline (Steps 0-6)"""
        print("\n" + "="*70)
        print("🔧 STEP 1: Preprocessing Pipeline")
        print("="*70)
        
        questions = data['questions']
        answers = data['answers']
        languages = data['languages']
        
        # Run full preprocessing pipeline
        preproc_results = self.preprocessor.run_full_pipeline(
            questions['English'],  # Pass English questions as reference
            answers,
            languages
        )
        
        # Store results
        self.results['preprocessing'] = preproc_results
        self.results['embeddings'] = preproc_results['embeddings']
        self.results['joint_embeddings'] = preproc_results['joint_embeddings']
        self.results['joint_labels'] = preproc_results['joint_labels']
        self.results['tokenisation_parity'] = preproc_results['tokenisation_parity']
        self.results['corpus_statistics'] = preproc_results['corpus_statistics']
        
        print(f"\n Preprocessing complete!")
        print(f"   Joint embedding space: {preproc_results['joint_embeddings'].shape}")
        
        return self.results
    
    def run_statistical_audit(self, data: Dict) -> Dict:
        """Run Stratum I: Statistical bias audit"""
        print("\n" + "="*70)
        print("STEP 2: Stratum I - Statistical Bias Audit")
        print("="*70)
        
        questions = data['questions']
        answers = self.results['preprocessing']['normalised_texts']
        
        # Run full statistical audit
        stat_results = self.stat_auditor.run_full_audit(questions, answers)
        
        self.results['statistical'] = stat_results
        
        print(f"\n Statistical audit complete!")
        print(f"   Flags generated: {len(stat_results['flags'])}")
        
        return self.results
    
    def run_linguistic_audit(self, data: Dict) -> Dict:
        """Run Stratum II: Linguistic bias audit"""
        print("\n" + "="*70)
        print("STEP 3: Stratum II - Linguistic Bias Audit")
        print("="*70)
        
        questions = data['questions']
        answers = self.results['preprocessing']['normalised_texts']
        
        # Prepare tokeniser performances from preprocessing
        tokeniser_performances = {}
        tp_df = self.results['tokenisation_parity']
        if tp_df is not None and not tp_df.empty:
            for lang in data['languages']:
                lang_data = tp_df[tp_df['Language'] == lang]
                if not lang_data.empty:
                    tokeniser_performances[lang] = {
                        'mBERT': {'fertility': lang_data[lang_data['Tokeniser'] == 'mBERT']['Fertility_Penalty'].values[0] if not lang_data[lang_data['Tokeniser'] == 'mBERT'].empty else 1.0},
                        'XLM-R': {'fertility': lang_data[lang_data['Tokeniser'] == 'XLM-R']['Fertility_Penalty'].values[0] if not lang_data[lang_data['Tokeniser'] == 'XLM-R'].empty else 1.0},
                        'AfriBERTa': {'fertility': lang_data[lang_data['Tokeniser'] == 'AfriBERTa']['Fertility_Penalty'].values[0] if not lang_data[lang_data['Tokeniser'] == 'AfriBERTa'].empty else 1.0}
                    }
        
        # Prepare sample words and tokeniser segmentations
        sample_words = self.preprocessor._get_sample_words(answers)
        tokeniser_segmentations = {}
        for lang in data['languages']:
            tokeniser_segmentations[lang] = {}
            for word in sample_words.get(lang, []):
                # Simulate tokeniser segmentations
                tokeniser_segmentations[lang][word] = self.preprocessor._simulate_tokenise(word, 1.5)
        
        # Run full linguistic audit
        ling_results = self.ling_auditor.run_full_audit(
            questions, answers, tokeniser_performances, 
            sample_words, tokeniser_segmentations
        )
        
        self.results['linguistic'] = ling_results
        
        print(f"\n Linguistic audit complete!")
        print(f"   Flags generated: {len(ling_results['flags'])}")
        
        return self.results
    
    def run_model_audit(self, data: Dict) -> Dict:
        """Run Stratum III: Model bias audit"""
        print("\n" + "="*70)
        print("STEP 4: Stratum III - Model Bias Audit")
        print("="*70)
        
        questions = data['questions']
        answers = data['answers']  # Use original answers, not normalized
        
        # Run full model audit
        model_results = self.model_auditor.run_full_audit(questions, answers)
        
        self.results['model'] = model_results
        
        print(f"\n Model audit complete!")
        print(f"   Flags generated: {len(model_results['flags'])}")
        
        return self.results
    
    def run_cross_lingual_evaluation(self, data: Dict) -> Dict:
        """Run cross-lingual evaluation engine"""
        print("\n" + "="*70)
        print("STEP 5: Cross-Lingual Evaluation Engine")
        print("="*70)
        
        embeddings = self.results['embeddings']
        questions = data['questions']
        topics = data.get('topics', [])
        
        # Run full cross-lingual evaluation
        cl_results = self.cross_lingual.run_full_evaluation(embeddings, questions, topics)
        
        self.results['cross_lingual'] = cl_results
        
        print(f"\n Cross-lingual evaluation complete!")
        print(f"   RCA cases: {len(cl_results['rca_results'])}")
        print(f"   Flags generated: {len(cl_results['flags'])}")
        
        return self.results
    
    def create_visualizations(self, data: Dict) -> Dict:
        """Create all visualizations for the pipeline"""
        print("\n" + "="*70)
        print("STEP 6: Creating Visualizations")
        print("="*70)
        
        # Extract data for visualizations
        sdi_matrix = self.results['cross_lingual'].get('sdi_matrix')
        tp_df = self.results['tokenisation_parity']
        trust_results = self.results['linguistic'].get('trust_aware_results')
        rca_results = self.results['cross_lingual'].get('rca_results')
        bias_patterns = self.results['cross_lingual'].get('bias_patterns')
        performance_df = self.results['model'].get('performance_results')
        
        # Create visualizations
        if sdi_matrix is not None:
            self.viz_dashboard.plot_sdi_heatmap(sdi_matrix)
        
        if tp_df is not None:
            self.viz_dashboard.plot_tokenisation_parity(tp_df)
        
        if trust_results:
            self.viz_dashboard.plot_trust_aware_results(trust_results)
        
        if rca_results:
            self.viz_dashboard.plot_rca_summary(rca_results)
        
        if bias_patterns is not None:
            self.viz_dashboard.plot_bias_pattern_heatmap(bias_patterns)
        
        if performance_df is not None:
            self.viz_dashboard.plot_performance_comparison(performance_df)
        
        # Create 3D embedding visualization
        joint_embeddings = self.results['joint_embeddings']
        joint_labels = self.results['joint_labels']
        if len(joint_embeddings) > 0:
            self.viz_dashboard.plot_3d_embedding_space(
                joint_embeddings, joint_labels, 
                title="Multilingual Maternal Health Answer Embeddings"
            )
        
        # Create interrogative analysis visualization
        inter_df = self.results['linguistic'].get('interrogative_analysis')
        if inter_df is not None:
            self.viz_dashboard.plot_interrogative_analysis(inter_df)
        
        # Create comprehensive dashboard
        self.viz_dashboard.create_comprehensive_dashboard(
            sdi_matrix=sdi_matrix,
            tp_df=tp_df,
            performance_df=performance_df,
            trust_results=trust_results,
            rca_results=rca_results,
            bias_patterns=bias_patterns
        )
        
        print(f"\n Visualizations complete! Saved to {FIGURES_DIR}")
        
        return self.results
    
    def take_bias_snapshot(self, data: Dict, iteration_name: str):
        """Take a bias snapshot for tracking"""
        print(f"\n Taking bias snapshot: {iteration_name}")
        
        # Extract metrics for snapshot
        sdi_matrix = self.results['cross_lingual'].get('sdi_matrix')
        performance_df = self.results['model'].get('performance_results')
        tp_df = self.results['tokenisation_parity']
        morph_results = self.results['linguistic'].get('morphological_alignment', {})
        joint_embeddings = self.results['joint_embeddings']
        joint_labels = self.results['joint_labels']
        
        # Get topics
        topics = data.get('topics', ['general'] * 5)
        # Extend topics to match embedding count
        if len(joint_embeddings) > 0:
            extended_topics = []
            for i in range(len(joint_embeddings)):
                extended_topics.append(topics[i % len(topics)])
        else:
            extended_topics = []
        
        # Create snapshot
        snapshot = self.bias_tracker.take_snapshot(
            intervention_description=iteration_name,
            sdi_matrix=sdi_matrix if sdi_matrix is not None else pd.DataFrame(),
            performance_df=performance_df if performance_df is not None else pd.DataFrame(),
            tokenisation_df=tp_df if tp_df is not None else pd.DataFrame(),
            morph_results=morph_results,
            embeddings=joint_embeddings,
            language_labels=joint_labels,
            topic_labels=extended_topics
        )
        
        self.results['current_snapshot'] = snapshot
        
        return snapshot
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive final bias report"""
        print("\n" + "="*70)
        print("Generating Final Report")
        print("="*70)
        
        # Collect all flags
        all_flags = []
        all_flags.extend(self.results.get('statistical', {}).get('flags', []))
        all_flags.extend(self.results.get('linguistic', {}).get('flags', []))
        all_flags.extend(self.results.get('model', {}).get('flags', []))
        all_flags.extend(self.results.get('cross_lingual', {}).get('flags', []))
        
        # Calculate key metrics
        sdi_matrix = self.results.get('cross_lingual', {}).get('sdi_matrix')
        avg_sdi = None
        if sdi_matrix is not None and not sdi_matrix.empty:
            upper_tri = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)]
            avg_sdi = np.mean(upper_tri) if len(upper_tri) > 0 else None
        
        # Collect cultural preservation items
        cultural_preserve = []
        for result in self.results.get('cross_lingual', {}).get('rca_results', []):
            if result.preserve:
                cultural_preserve.append(result.recommendation)
        
        # Generate report
        report = {
            'experiment_name': EXPERIMENT_NAME,
            'version': EXPERIMENT_VERSION,
            'execution_time': datetime.now().isoformat(),
            'languages_analyzed': PRIMARY_LANGUAGES,
            'n_questions': self.results.get('preprocessing', {}).get('metadata', {}).n_questions if self.results.get('preprocessing', {}).get('metadata') else 5,
            'key_metrics': {
                'average_sdi': float(avg_sdi) if avg_sdi is not None else None,
                'sdi_interpretation': (
                    'High bias detected' if avg_sdi and avg_sdi > THRESHOLDS['sdi_high'] else
                    'Moderate bias' if avg_sdi and avg_sdi > THRESHOLDS['sdi_moderate'] else
                    'Low bias - good alignment' if avg_sdi else 'Not computed'
                ),
                'total_flags': len(all_flags)
            },
            'flags': all_flags[:20],  # Limit to 20 flags
            'cultural_preservation': cultural_preserve[:10],  # Limit to 10 items
            'bias_tracker_summary': self.bias_tracker.get_summary()
        }
        
        # Save report
        report_path = os.path.join(REPORTS_DIR, f"final_report_{EXPERIMENT_VERSION}.json")
        save_report(report, report_path)
        
        # Print summary
        print("\n" + "-"*40)
        print(" EXECUTIVE SUMMARY")
        print("-"*40)
        
        print(f"\n Languages: {', '.join(PRIMARY_LANGUAGES)}")
        print(f" Questions: {report['n_questions']}")
        
        print(f"\n Bias Detection:")
        if report['key_metrics']['average_sdi']:
            print(f"   • Average SDI: {report['key_metrics']['average_sdi']:.3f}")
        print(f"   • {report['key_metrics']['sdi_interpretation']}")
        print(f"   • Total Issues Detected: {report['key_metrics']['total_flags']}")
        
        if cultural_preserve:
            print(f"\n Cultural Knowledge to PRESERVE: {len(cultural_preserve)} items")
            for item in cultural_preserve[:3]:
                print(f"   • {item[:80]}...")
        
        print(f"\n Full report saved to: {report_path}")
        print(f"Figures saved to: {FIGURES_DIR}")
        print(f"Output directory: {OUTPUT_DIR}")
        
        return report
    
    def run(self) -> Dict:
        """
        Run the complete bias audit pipeline
        """
        print("\n" + "="*70)
        print(f" Starting {EXPERIMENT_NAME} v{EXPERIMENT_VERSION}")
        print("="*70)
        
        # Load data
        data = self.load_data()
        
        try:
            # Run all pipeline stages
            self.run_preprocessing(data)
            self.run_statistical_audit(data)
            self.run_linguistic_audit(data)
            self.run_model_audit(data)
            self.run_cross_lingual_evaluation(data)
            
            # Take baseline snapshot
            self.take_bias_snapshot(data, "Baseline - Complete Audit")
            
            # Create visualizations
            self.create_visualizations(data)
            
        except Exception as e:
            print(f"\n Warning during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate final report
        final_report = self.generate_final_report()
        
        print("\n" + "="*70)
        print(f"{EXPERIMENT_NAME} v{EXPERIMENT_VERSION} Complete!")
        print("="*70)
        
        return final_report


# Main execution
if __name__ == "__main__":
    # Initialize and run pipeline
    pipeline = MaHealthBiasAuditPipeline(
        save_visuals=True,
        show_visuals=False  # Set to True to display figures interactively
    )
    
    results = pipeline.run()
    
    print("\n Bias audit completed successfully!")