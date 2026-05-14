"""
MaHealthBiasAudit v2 - Complete Pipeline Runner
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import (LANGUAGES, PRIMARY_LANGUAGES, THRESHOLDS, 
                   RANDOM_SEED, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR,
                   INTERROGATIVE_PATTERNS, CULTURAL_TERMINOLOGY, DIALECT_MARKERS)
from utils import (set_seed, normalize_text, classify_maternal_topic, 
                  compute_chrf)
from preprocessing import MultilingualPreprocessor
from stratum_i_statistical import StatisticalBiasAuditor
from stratum_ii_linguistic import LinguisticBiasAuditor
from cross_lingual_evaluation import CrossLingualEvaluator
from bias_tracker import BiasTracker
from visualization_dashboard import BiasVisualizationDashboard


class MaHealthBiasAuditPipeline:
    """
    Pipeline for Multilingual Maternal Health QA Bias Audit
    """
    
    def __init__(self, show_visuals: bool = True):
        """Initialize the complete pipeline"""
        set_seed(RANDOM_SEED)
        self.show_visuals = show_visuals
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(FIGURES_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Initialize all components
        print("\n" + "="*70)
        print(" MaHealthBiasAudit v2 - Initializing")
        print("="*70)
        
        self.preprocessor = MultilingualPreprocessor()
        self.stat_auditor = StatisticalBiasAuditor()
        self.ling_auditor = LinguisticBiasAuditor()
        self.cross_lingual = CrossLingualEvaluator()
        self.bias_tracker = BiasTracker(save_path=f"{REPORTS_DIR}/bias_history.json")
        self.viz_dashboard = BiasVisualizationDashboard(
            save_figures=True, 
            output_dir=FIGURES_DIR,
            show_display=show_visuals
        )
        
        # Storage for results
        self.results = {}
        
        print("\n All components initialized successfully")
        print(f"   Visualizations will {'be shown' if show_visuals else 'NOT be shown'} interactively")
    
    def load_data(self) -> Dict:
        """Loading the maternal health QA dataset"""
        # Core questions
        maternal_health_questions = [
            "What are the essential nutrients a pregnant woman should consume daily?",
            "What are the common signs of labor, and when should a pregnant woman seek medical attention?",
            "What are the benefits of breastfeeding for both the mother and the baby?",
            "How can a new mother cope with postpartum depression, and what support systems are available?",
            "What are the recommended vaccinations for a child from birth to one year of age, and why are they important?"
        ]
        
        # Answers in each language
        answers = {
            'English': [
                "A pregnant woman should consume folic acid, iron, calcium, protein, iodine, and omega-3 fatty acids daily. These nutrients support fetal brain development, prevent anemia, strengthen bones, and promote overall maternal health.",
                "Common signs include regular contractions, lower back pain, water breaking, and cervical dilation. Medical attention should be sought when contractions are frequent (every 5 minutes), if there is bleeding, or reduced fetal movement.",
                "Breastfeeding provides optimal nutrition, strengthens the baby's immune system, reduces risk of SIDS, and promotes bonding. For mothers, it reduces risk of breast cancer, ovarian cancer, and helps with postpartum weight loss.",
                "New mothers can cope with postpartum depression through counseling, support groups, medication if needed, and social support from family. Community health workers can provide referrals to mental health services.",
                "Recommended vaccinations include BCG at birth, Polio (OPV/IPV) at birth, 6, 10, 14 weeks, Pentavalent at 6, 10, 14 weeks, Pneumococcal at 6, 10, 14 weeks, Rotavirus at 6, 10 weeks, and Measles at 9 months."
            ],
            'Swahili': [
                "Mwanamke mjamzito anapaswa kula virutubisho muhimu kama asidi ya foliki, chuma, kalsiamu, protini, iodini, na mafuta ya omega-3 kila siku. Virutubisho hivi husaidia ukuaji wa ubongo wa mtoto, kuzuia upungufu wa damu, na kuimarisha afya ya mama.",
                "Dalili za uchungu wa kujifungua ni pamoja na mikazo ya mara kwa mara, maumivu ya mgongo, kupasuka kwa maji ya uzazi, na kufunguka kwa mlango wa kizazi. Huduma ya afya inahitajika ikiwa mikazo ni ya mara kwa mara au kuna damu au mtoto hasogei.",
                "Kunyonyesha hutoa lishe bora, huimarisha kinga ya mtoto, hupunguza hatari ya SIDS, na huongeza uhusiano. Kwa mama, hupunguza hatari ya saratani ya matiti, saratani ya ovari, na husaidia katika kupunguza uzito baada ya kujifungua.",
                "Akina mama wanaweza kukabiliana na mfadhaiko baada ya kujifungua kupitia ushauri nasaha, vikundi vya msaada, dawa ikiwa ni lazima, na msaada wa kijamii kutoka kwa familia. Wahudumu wa afya ya jamii wanaweza kutoa rufaa kwa huduma za afya ya akili.",
                "Chanjo zinazopendekezwa ni pamoja na BCG kuzaliwa, Polio (OPV/IPV) kuzaliwa, wiki 6, 10, 14, Pentavalent wiki 6, 10, 14, Pneumococcal wiki 6, 10, 14, Rotavirus wiki 6, 10, na Surua miezi 9."
            ],
            'Luganda': [
                "Omukyala embuto alina okulya folic acid, ekyuma, kalisiyamu, omugaati, ayodini, na mafuta ga omega-3 buli lunaku. Ebyetaago bino biyamba okukula kwa bwana, kuziyiza omusujja gw'ekyuma, okunyweza amagumba, n'okulongoosa obulamu bw'omukyala.",
                "Obubonero bw'okuzala mulimu okuluma okwewalula, okuluma mu mugongo, amazzi okukulukuta, n'okugaziya k'endabirwamu. Obujanjabi bweetaaga ng'okuluma kuba kwa buli dakika ttaano, oba omusaayi gukulukuta, oba omwana akayonja.",
                "Okunyonsa kubera omwana eby'okulya ebirungi, kuzimba obudde bw'obutaasa bw'omwana, kukendeeza akabi ke SIDS, era kuwa amaka ag'omukwano. Kwa maama, kukendeeza akabi k'kansa y'abeera, kansa ey'ovari, era kuyamba okukendeeza obuzito oluvanyuma lw'okuzala.",
                "Baama abapya bayinza okuwangula okweraliikirira oluvanyuma lw'okuzala nga bakozesa okubuulirirwa, ebibiina ebiyambako, eddagala bwe kiba nga kyetaagisa, n'obuyambi obuva mu maka. Abavumirirwa by'obulamu mu kitundu bayinza okuwa obujanjabi bwa by'obulamu bw'omuntu.",
                "Engatto ez'okwetegeka zirimu BCG ku kuzalibwa, Polio (OPV/IPV) ku kuzalibwa, wiiki 6, 10, 14, Pentavalent wiiki 6, 10, 14, Pneumococcal wiiki 6, 10, 14, Rotavirus wiiki 6, 10, ne Kasipi mu miezi 9."
            ],
            'Runyankore': [
                "Omukazi embuto ata hairwe okurya folic acid, ekyoma, kalisiyamu, omugisha, ayodini, n'amavuta ga omega-3 buzoba. Ebirikukozesa binu nibikozesa okukura kw'omwana, okuziyiza omushwiju, okukomeza amagufa, n'okureeza obulamu bw'omukazi.",
                "Obubonero bw'okuzaara nikwataho okubaba okwera, okubaba omugongo, amaizi kweijuka, n'okugaziya endabirwamu. Obujanjabi nibukozesa ahabw'okubaba okwera buli dakika ttaano, omusaayi kuguruka, oba omwana kutagenda.",
                "Okunyonsa nikugaba eby'okurya ebirungi, nikukomeza obutaasa bw'omwana, nikukendeeza eky'akabi SIDS, kandi nikukozesa kwebiragiro. Ahabw'omukazi, nikukendeeza akabi k'kansa y'abeera, kansa y'ovari, kandi nikukozesa mukukendeeza oburemere bw'omukazi.",
                "Abazaire abashya basinika kujwana eky'orugyo ahabw'okwejinja omu kwetantara, amahema abambuzi, omwenda nikiba nikisemeriire, n'obubazi oburuga omu muryango. Abavugyiri b'obujanjabi omu kicweka basinika kuha obujanjabi bw'obwire bw'omutima.",
                "Enyaana ezikwataho nikuba BCG omu kuzaarwa, Polio (OPV/IPV) omu kuzaarwa, sabiiti 6, 10, 14, Pentavalent sabiiti 6, 10, 14, Pneumococcal sabiiti 6, 10, 14, Rotavirus sabiiti 6, 10, n'Omushija omwezi 9."
            ]
        }
        
        # Structure data by language
        questions_by_lang = {lang: maternal_health_questions for lang in PRIMARY_LANGUAGES}
        answers_by_lang = {lang: answers.get(lang, answers['English']) for lang in PRIMARY_LANGUAGES}
        
        # Add topic classification for each question
        topics = [classify_maternal_topic(q) for q in maternal_health_questions]
        
        print(f"\n📊 Data loaded:")
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
        """Run preprocessing pipeline"""
        print("\n" + "="*60)
        print(" STEP 1: Preprocessing & NLP Pipeline")
        print("="*60)
        
        questions = data['questions']
        answers = data['answers']
        languages = data['languages']
        
        all_embeddings = {}
        tokeniser_performances = {t: {} for t in ['mBERT', 'XLM-R', 'AfriBERTa']}
        
        for lang in languages:
            print(f"\n  Processing {lang}...")
            
            a_texts = answers.get(lang, [])
            eng_texts = answers.get('English', [])
            
            # Step 2: Text Normalisation
            norm_answers = [self.preprocessor.text_normalisation(a, lang) for a in a_texts]
            
            # Step 3: Tokenisation analysis
            token_results = self.preprocessor.multi_tokeniser_analysis(norm_answers, lang, eng_texts)
            
            # Store tokeniser performances
            for tokeniser in ['mBERT', 'XLM-R', 'AfriBERTa']:
                fertility = np.mean(token_results[tokeniser]['fertility']) if token_results[tokeniser]['fertility'] else 1.0
                tokeniser_performances[tokeniser][lang] = fertility
            
            # Step 5: Sentence Embedding
            embeddings = self.preprocessor.generate_embeddings(norm_answers, model_name='LaBSE')
            all_embeddings[lang] = embeddings
            
            print(f"    ✓ Embedded {len(embeddings)} answers (shape: {embeddings.shape})")
        
        # Create joint embedding space
        joint_embeddings, joint_labels = self.preprocessor.create_joint_embedding_space(all_embeddings)
        
        # Compute tokenisation parity
        tp_df = self.preprocessor.compute_tokenisation_parity()
        
        self.results['embeddings'] = all_embeddings
        self.results['joint_embeddings'] = joint_embeddings
        self.results['joint_labels'] = joint_labels
        self.results['tokenisation_parity'] = tp_df
        self.results['tokeniser_performances'] = tokeniser_performances
        
        print(f"\n  ✓ Joint embedding space: {joint_embeddings.shape[0]} vectors")
        
        return self.results
    
    def run_statistical_audit(self, data: Dict) -> Dict:
        """Run Stratum I: Statistical bias audit"""
        print("\n" + "="*60)
        print(" STEP 2: Stratum I - Statistical Bias Audit")
        print("="*60)
        
        questions = data['questions']
        answers = data['answers']
        
        # Compute corpus statistics
        stats = self.stat_auditor.compute_corpus_statistics(questions, answers)
        
        print("\n  Corpus Statistics:")
        for lang, stat in stats.items():
            print(f"    {lang}: TTR={stat.type_token_ratio:.3f}, "
                  f"Vocab={stat.vocab_size}, Q_len={stat.avg_question_length:.1f}")
        
        # Compute JSD
        jsd_results = self.stat_auditor.compute_jensen_shannon_divergence(questions)
        
        if jsd_results and 'average' in jsd_results:
            print(f"\n  Average Jensen-Shannon Divergence: {jsd_results['average']:.4f}")
        
        # Generate report
        report_df = self.stat_auditor.generate_bias_report()
        flags = self.stat_auditor.get_flags()
        
        self.results['statistical_report'] = report_df
        self.results['statistical_flags'] = flags
        self.results['jsd_results'] = jsd_results
        
        print(f"\n  ✓ Statistical audit complete. Flags: {len(flags)}")
        
        return self.results
    
    def run_linguistic_audit(self, data: Dict) -> Dict:
        """Run Stratum II: Linguistic bias audit"""
        print("\n" + "="*60)
        print(" STEP 3: Stratum II - Linguistic Bias Audit")
        print("="*60)
        
        questions = data['questions']
        answers = data['answers']
        tokeniser_perf = self.results.get('tokeniser_performances', {})
        
        # Compute tokenisation parity
        tp_df = self.ling_auditor.compute_tokenisation_parity(questions, tokeniser_perf)
        
        print("\n  Tokenisation Parity (Fertility Penalty):")
        for _, row in tp_df.iterrows():
            flag = "HIGH" if row['TP_Flag'] else "✓"
            print(f"    {flag} {row['Language']} ({row['Tokeniser']}): {row['Fertility_Penalty']:.2f}")
        
        # Analyze interrogative structure
        inter_df = self.ling_auditor.analyze_interrogative_structure(questions)
        self.viz_dashboard.plot_interrogative_analysis(inter_df)
        
        print("\n  Interrogative Structure Analysis:")
        mismatches = inter_df[inter_df['Mismatch']]
        if not mismatches.empty:
            print(f"  {len(mismatches)} questions have non-English patterns")
        
        # Trust-Aware Module
        print("\n  Trust-Aware Module (Cultural Terminology):")
        for lang in data['languages']:
            sample_text = ' '.join(answers.get(lang, [''])[:2])
            trust_result = self.ling_auditor.trust_aware_module([sample_text], lang)
            
            status = "✓ PRESERVE" if trust_result.preservation_needed else "ℹ️"
            print(f"    {status} {lang}: Trust Score={trust_result.trust_score:.2f}, "
                  f"Terms found={len(trust_result.cultural_terms_found)}")
        
        self.viz_dashboard.plot_trust_aware_results(self.ling_auditor.trust_results)
        
        # Compute morphological alignment
        mas_df = self.ling_auditor.compute_morphological_alignment_scores()
        flags = self.ling_auditor.get_flags()
        
        self.results['linguistic_tp'] = tp_df
        self.results['interrogative_analysis'] = inter_df
        self.results['trust_results'] = self.ling_auditor.trust_results
        self.results['linguistic_flags'] = flags
        self.results['morphological_alignment'] = mas_df
        
        print(f"\n  ✓ Linguistic audit complete. Flags: {len(flags)}")
        
        return self.results
    
    def run_cross_lingual_evaluation(self, data: Dict) -> Dict:
        """Run cross-lingual evaluation engine"""
        print("\n" + "="*60)
        print(" STEP 4: Cross-Lingual Evaluation Engine")
        print("="*60)
        
        embeddings = self.results.get('embeddings', {})
        questions = data['questions']
        topics = data.get('topics', [])
        
        if not embeddings:
            print("  No embeddings found. Run preprocessing first.")
            return self.results
        
        # Compute SDI
        sdi_matrix = self.cross_lingual.compute_semantic_divergence_index(embeddings, questions)
        self.viz_dashboard.plot_sdi_heatmap(sdi_matrix)
        
        print("\n  Semantic Divergence Index (SDI):")
        for i, lang1 in enumerate(sdi_matrix.index):
            for j, lang2 in enumerate(sdi_matrix.columns):
                if i < j:
                    sdi = sdi_matrix.loc[lang1, lang2]
                    severity = "HIGH" if sdi > THRESHOLDS['sdi_high'] else \
                              "MODERATE" if sdi > THRESHOLDS['sdi_moderate'] else " LOW"
                    print(f"    {severity} {lang1} ↔ {lang2}: {sdi:.3f}")
        
        # Compute MRR
        mrr_matrix = self.cross_lingual.compute_mean_reciprocal_rank(embeddings)
        
        # Compute cluster purity
        joint_embeddings = self.results.get('joint_embeddings')
        joint_labels = self.results.get('joint_labels')
        
        if joint_embeddings is not None and len(joint_embeddings) > 0:
            lang_purity = self.cross_lingual.compute_cluster_purity(joint_embeddings, joint_labels)
            print(f"\n  Language Cluster Purity: {lang_purity:.3f}")
            print(f"    → {'BIAS: embeddings cluster by language' if lang_purity > 0.6 else '✓ Good cross-lingual alignment'}")
        
        # Root Cause Attribution
        rca_results = self.cross_lingual.root_cause_attribution_cascade()
        self.viz_dashboard.plot_rca_summary(rca_results)
        
        print("\n  Root Cause Attribution (RCA) Cascade:")
        cultural_preserve = []
        for result in rca_results:
            if result.preserve:
                cultural_preserve.append(result)
            elif result.confidence > 0.7:
                print(f"    → {result.root_cause}: {result.language_pair[0]}-{result.language_pair[1]} "
                      f"(conf={result.confidence:.2f})")
        
        if cultural_preserve:
            print(f"\n  🌍 Cultural Knowledge to PRESERVE ({len(cultural_preserve)} items)")
            for result in cultural_preserve[:3]:
                print(f"    → {result.recommendation}")
        
        # Detect bias patterns by topic
        if topics and len(topics) > 0:
            extended_topics = []
            for lang in embeddings.keys():
                extended_topics.extend(topics[:len(embeddings[lang])])
            
            bias_patterns = self.cross_lingual.detect_bias_patterns(embeddings, questions, extended_topics)
            self.viz_dashboard.plot_bias_pattern_heatmap(bias_patterns)
            self.viz_dashboard.bias_patterns = bias_patterns
            
            print("\n  Bias Patterns by Maternal Health Topic:")
            for _, row in bias_patterns.iterrows():
                severity = "🔴" if row['bias_severity'] == 'high' else "🟡" if row['bias_severity'] == 'moderate' else "🟢"
                print(f"    {severity} {row['topic']}: SDI={row['avg_sdi']:.3f} ({row['bias_severity']})")
            
            self.results['bias_patterns'] = bias_patterns
        
        # Generate report
        report = self.cross_lingual.generate_report()
        flags = self.cross_lingual.get_flags()
        
        self.results['sdi_matrix'] = sdi_matrix
        self.results['mrr_matrix'] = mrr_matrix
        self.results['rca_results'] = rca_results
        self.results['cross_lingual_report'] = report
        self.results['cross_lingual_flags'] = flags
        self.results['lang_cluster_purity'] = lang_purity if 'lang_purity' in dir() else 0.5
        
        print(f"\n  ✓ Cross-lingual evaluation complete. RCA: {len(rca_results)} issues analyzed")
        
        return self.results
    
    def create_visualizations(self) -> Dict:
        """Create all visualizations for the pipeline"""
        print("\n" + "="*60)
        print(" STEP 5: Creating Visualizations")
        print("="*60)
        
        # 3D embedding space visualization
        joint_embeddings = self.results.get('joint_embeddings')
        joint_labels = self.results.get('joint_labels')
        
        fig_3d = None
        if joint_embeddings is not None and len(joint_embeddings) > 0:
            # Create topic labels
            available_topics = ['antenatal_care', 'labor_delivery', 'postnatal_care', 
                               'mental_health', 'child_health', 'nutrition']
            topics = []
            for i in range(len(joint_embeddings)):
                topics.append(available_topics[i % len(available_topics)])
            
            fig_3d = self.viz_dashboard.plot_3d_embedding_space(
                joint_embeddings, joint_labels, topics,
                title="Multilingual Maternal Health Answer Embeddings"
            )
        
        # Tokenisation parity
        tp_df = self.results.get('tokenisation_parity')
        if tp_df is not None and not tp_df.empty:
            self.viz_dashboard.plot_tokenisation_parity(tp_df)
        
        # Create complete dashboard
        self.viz_dashboard.create_dashboard(
            sdi_matrix=self.results.get('sdi_matrix', pd.DataFrame()),
            tp_df=self.results.get('tokenisation_parity', pd.DataFrame()),
            performance_df=pd.DataFrame(),
            morph_results=self.results.get('morphological_alignment', {}),
            inter_df=self.results.get('interrogative_analysis'),
            trust_results=self.results.get('trust_results'),
            rca_results=self.results.get('rca_results'),
            embed_figure=fig_3d
        )
        
        print("\n  ✓ All visualizations created and displayed!")
        
        return self.results
    
    def generate_final_report(self) -> Dict:
        """Generate comprehensive final bias report"""
        print("\n" + "="*60)
        print(" FINAL REPORT: MaHealthBiasAudit v2")
        print("="*60)
        
        # Calculate key metrics
        sdi_matrix = self.results.get('sdi_matrix')
        avg_sdi = None
        if sdi_matrix is not None and not sdi_matrix.empty:
            upper_tri = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)]
            avg_sdi = np.mean(upper_tri) if len(upper_tri) > 0 else None
        
        # Compile report
        report = {
            'pipeline_version': '2.0',
            'execution_time': datetime.now().isoformat(),
            'languages_analyzed': self.results.get('languages', PRIMARY_LANGUAGES),
            'n_questions': self.results.get('n_questions', 5),
            'key_metrics': {
                'average_sdi': avg_sdi,
                'sdi_interpretation': (
                    'High bias detected' if avg_sdi and avg_sdi > THRESHOLDS['sdi_high'] else
                    'Moderate bias' if avg_sdi and avg_sdi > THRESHOLDS['sdi_moderate'] else
                    'Low bias - good alignment' if avg_sdi else 'Not computed'
                ),
                'lang_cluster_purity': self.results.get('lang_cluster_purity', 0.5)
            },
            'flags': {
                'statistical': self.results.get('statistical_flags', []),
                'linguistic': self.results.get('linguistic_flags', []),
                'cross_lingual': self.results.get('cross_lingual_flags', [])
            },
            'cultural_preservation': [
                r.recommendation for r in self.results.get('rca_results', []) 
                if r.preserve
            ]
        }
        
        # Print summary
        print("\n" + "-"*40)
        print(" EXECUTIVE SUMMARY")
        print("-"*40)
        
        print(f"\n Languages: {', '.join(report['languages_analyzed'])}")
        print(f" Questions: {report['n_questions']}")
        
        print(f"\n Bias Detection:")
        if avg_sdi:
            print(f"   • Average SDI: {avg_sdi:.3f}")
        print(f"   • {report['key_metrics']['sdi_interpretation']}")
        print(f"   • Language Cluster Purity: {report['key_metrics']['lang_cluster_purity']:.3f}")
        
        total_flags = sum(len(f) for f in report['flags'].values())
        print(f"\n Issues Detected: {total_flags}")
        
        cultural_items = report['cultural_preservation']
        if cultural_items:
            print(f"\n Cultural Knowledge to PRESERVE: {len(cultural_items)} items")
            for item in cultural_items[:3]:
                print(f"   • {item[:80]}...")
        
        # Save report
        import json
        report_path = f"{REPORTS_DIR}/final_report.json"
        report_serializable = {
            k: v for k, v in report.items() 
            if isinstance(v, (str, int, float, list, dict, type(None)))
        }
        with open(report_path, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print(f"\n Full report saved to {report_path}")
        print(f" Figures saved to {FIGURES_DIR}/")
        
        return report
    
    def run(self) -> Dict:
        """Run the complete pipeline"""
        print("\n" + "="*70)
        print(" MaHealthBiasAudit v2 - Complete Pipeline Execution")
        print("="*70)
        
        # Load data
        data = self.load_data()
        
        try:
            # Run all pipeline stages
            self.run_preprocessing(data)
            self.run_statistical_audit(data)
            self.run_linguistic_audit(data)
            self.run_cross_lingual_evaluation(data)
            self.create_visualizations()
        except Exception as e:
            print(f"\n Warning during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
        
        # Generate final report
        final_report = self.generate_final_report()
        
        print("\n" + "="*70)
        print(" Pipeline Complete!")
        print(f" Results saved to '{OUTPUT_DIR}/'")
        print("="*70)
        
        return final_report


# Main execution
if __name__ == "__main__":
    # Set show_visuals=True to display all figures interactively
    pipeline = MaHealthBiasAuditPipeline(show_visuals=True)
    results = pipeline.run()
    
    print("\n Visualizations have been generated and saved.")