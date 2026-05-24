#!/usr/bin/env python3
"""
MaHealthBiasAudit v2 - Runner Script
Simplified entry point for running the bias audit pipeline
"""

import argparse
import sys
import os
from datetime import datetime

def main():
    """Main entry point for bias audit runner"""
    
    parser = argparse.ArgumentParser(
        description='MaHealthBiasAudit v2 - Multilingual Maternal Health Bias Detection Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline with visualizations
    python run_bias_audit.py --full --save-viz
    
    # Run only preprocessing
    python run_bias_audit.py --preprocessing
    
    # Run with specific languages
    python run_bias_audit.py --full --languages English Swahili Luganda
    
    # Generate report only
    python run_bias_audit.py --report
        """
    )
    
    parser.add_argument('--full', action='store_true', 
                       help='Run full bias audit pipeline')
    parser.add_argument('--preprocessing', action='store_true',
                       help='Run only preprocessing step')
    parser.add_argument('--statistical', action='store_true',
                       help='Run only statistical bias audit')
    parser.add_argument('--linguistic', action='store_true',
                       help='Run only linguistic bias audit')
    parser.add_argument('--model', action='store_true',
                       help='Run only model bias audit')
    parser.add_argument('--cross-lingual', action='store_true',
                       help='Run only cross-lingual evaluation')
    parser.add_argument('--report', action='store_true',
                       help='Generate report from existing results')
    
    parser.add_argument('--languages', nargs='+', default=['English', 'Swahili', 'Yoruba', 'Amharic'],
                       help='Languages to analyze (default: all 4 primary languages)')
    parser.add_argument('--save-viz', action='store_true', default=True,
                       help='Save visualization figures')
    parser.add_argument('--show-viz', action='store_true',
                       help='Show visualization figures interactively')
    parser.add_argument('--output-dir', default='mahealth_bias_output',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # If no specific step selected, run full pipeline
    if not any([args.full, args.preprocessing, args.statistical, 
                args.linguistic, args.model, args.cross_lingual, args.report]):
        args.full = True
    
    # Set random seed
    import numpy as np
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Import config FIRST to get default values
    import config
    
    # Update output directory if specified (before importing other modules)
    if args.output_dir != 'mahealth_bias_output':
        config.OUTPUT_DIR = args.output_dir
        config.FIGURES_DIR = f"{args.output_dir}/figures"
        config.REPORTS_DIR = f"{args.output_dir}/reports"
        config.MODELS_DIR = f"{args.output_dir}/models"
        config.LOGS_DIR = f"{args.output_dir}/logs"
        
        # Recreate directories
        for dir_path in [config.OUTPUT_DIR, config.FIGURES_DIR, 
                        config.REPORTS_DIR, config.MODELS_DIR, config.LOGS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    # Now import other modules that depend on config
    from main import MaHealthBiasAuditPipeline
    
    # Override languages if specified
    if args.languages != config.PRIMARY_LANGUAGES:
        print(f"📝 Using custom language set: {args.languages}")
        config.PRIMARY_LANGUAGES = args.languages
    
    print("\n" + "="*70)
    print(" MaHealthBiasAudit v2 - Bias Detection Pipeline")
    print("="*70)
    print(f" Execution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Random seed: {args.seed}")
    print(f" Output directory: {config.OUTPUT_DIR}")
    print("="*70)
    
    if args.full:
        print("\n🚀 Running FULL bias audit pipeline...")
        pipeline = MaHealthBiasAuditPipeline(
            save_visuals=args.save_viz,
            show_visuals=args.show_viz
        )
        results = pipeline.run()
        
    elif args.preprocessing:
        print("\n🔧 Running only PREPROCESSING...")
        from preprocessing import MultilingualPreprocessor
        
        # Load data using pipeline helper
        temp_pipeline = MaHealthBiasAuditPipeline(save_visuals=False, show_visuals=False)
        data = temp_pipeline.load_data()
        
        preprocessor = MultilingualPreprocessor()
        results = preprocessor.run_full_pipeline(
            data['questions']['English'],
            data['answers'],
            data['languages']
        )
        
        print("\n📊 Preprocessing Results:")
        print(f"   Metadata: {results['metadata'].name}")
        print(f"   Tokenisation Parity: {results['tokenisation_parity'].shape}")
        print(f"   Joint Embeddings: {results['joint_embeddings'].shape}")
        
    elif args.statistical:
        print("\n📊 Running STATISTICAL bias audit...")
        from stratum_i_statistical import StatisticalBiasAuditor
        from preprocessing import MultilingualPreprocessor
        
        temp_pipeline = MaHealthBiasAuditPipeline(save_visuals=False, show_visuals=False)
        data = temp_pipeline.load_data()
        
        # Need normalized texts
        preprocessor = MultilingualPreprocessor()
        normalized = preprocessor.step2_text_normalisation(data['answers'])
        
        auditor = StatisticalBiasAuditor()
        results = auditor.run_full_audit(data['questions'], normalized)
        
        print("\n📊 Statistical Audit Results:")
        print(f"   Flags: {len(results['flags'])}")
        if results.get('information_specific_divergence'):
            for isd in results['information_specific_divergence'][:3]:
                print(f"   {isd.language_pair[0]} ↔ {isd.language_pair[1]}: ISD={isd.isd_value:.3f}")
        
    elif args.linguistic:
        print("\n🔤 Running LINGUISTIC bias audit...")
        from stratum_ii_linguistic import LinguisticBiasAuditor
        from preprocessing import MultilingualPreprocessor
        
        temp_pipeline = MaHealthBiasAuditPipeline(save_visuals=False, show_visuals=False)
        data = temp_pipeline.load_data()
        
        # Need normalized texts and tokeniser performances
        preprocessor = MultilingualPreprocessor()
        normalized = preprocessor.step2_text_normalisation(data['answers'])
        tp_df = preprocessor.step3_tokenisation_analysis(normalized)
        
        tokeniser_performances = {}
        for lang in data['languages']:
            lang_data = tp_df[tp_df['Language'] == lang]
            tokeniser_performances[lang] = {
                'mBERT': {'fertility': lang_data[lang_data['Tokeniser'] == 'mBERT']['Fertility_Penalty'].values[0] if not lang_data[lang_data['Tokeniser'] == 'mBERT'].empty else 1.0},
                'XLM-R': {'fertility': lang_data[lang_data['Tokeniser'] == 'XLM-R']['Fertility_Penalty'].values[0] if not lang_data[lang_data['Tokeniser'] == 'XLM-R'].empty else 1.0},
                'AfriBERTa': {'fertility': lang_data[lang_data['Tokeniser'] == 'AfriBERTa']['Fertility_Penalty'].values[0] if not lang_data[lang_data['Tokeniser'] == 'AfriBERTa'].empty else 1.0}
            }
        
        sample_words = preprocessor._get_sample_words(normalized)
        tokeniser_segmentations = {}
        for lang in data['languages']:
            tokeniser_segmentations[lang] = {}
            for word in sample_words.get(lang, []):
                tokeniser_segmentations[lang][word] = preprocessor._simulate_tokenise(word, 1.5)
        
        auditor = LinguisticBiasAuditor()
        results = auditor.run_full_audit(
            data['questions'], normalized, tokeniser_performances,
            sample_words, tokeniser_segmentations
        )
        
        print("\n🔤 Linguistic Audit Results:")
        print(f"   Flags: {len(results['flags'])}")
        if results.get('trust_aware_results'):
            for lang, trust in results['trust_aware_results'].items():
                print(f"   {lang}: Trust Score={trust.trust_score:.2f}, Terms={len(trust.cultural_terms_found)}")
        
    elif args.model:
        print("\n🤖 Running MODEL bias audit...")
        from stratum_iii_model import ModelBiasAuditor
        
        temp_pipeline = MaHealthBiasAuditPipeline(save_visuals=False, show_visuals=False)
        data = temp_pipeline.load_data()
        
        auditor = ModelBiasAuditor()
        results = auditor.run_full_audit(data['questions'], data['answers'])
        
        print("\n🤖 Model Audit Results:")
        print(f"   Performance records: {len(results['performance_results'])}")
        print(f"   Flags: {len(results['flags'])}")
        if not results['bias_metrics'].empty:
            print(results['bias_metrics'].to_string(index=False))
        
    elif args.cross_lingual:
        print("\n🌐 Running CROSS-LINGUAL evaluation...")
        from cross_lingual_evaluation import CrossLingualEvaluator
        from preprocessing import MultilingualPreprocessor
        
        temp_pipeline = MaHealthBiasAuditPipeline(save_visuals=False, show_visuals=False)
        data = temp_pipeline.load_data()
        
        # Need embeddings
        preprocessor = MultilingualPreprocessor()
        normalized = preprocessor.step2_text_normalisation(data['answers'])
        embeddings = preprocessor.step5_generate_embeddings(normalized)
        
        evaluator = CrossLingualEvaluator()
        results = evaluator.run_full_evaluation(embeddings, data['questions'], data['topics'])
        
        print("\n🌐 Cross-Lingual Results:")
        print(f"   SDI Matrix: {results['sdi_matrix'].shape}")
        print(f"   RCA cases: {len(results['rca_results'])}")
        print(f"   Flags: {len(results['flags'])}")
        
        if results['sdi_matrix'] is not None:
            print("\n   SDI Matrix:")
            print(results['sdi_matrix'].to_string())
        
    elif args.report:
        print("\n📋 Generating report from existing results...")
        import json
        
        # Find latest report
        report_files = [f for f in os.listdir(config.REPORTS_DIR) if f.startswith('final_report_') and f.endswith('.json')]
        if report_files:
            latest = max(report_files, key=lambda x: os.path.getctime(os.path.join(config.REPORTS_DIR, x)))
            with open(os.path.join(config.REPORTS_DIR, latest), 'r') as f:
                report = json.load(f)
            
            print("\n" + "="*60)
            print(" LATEST BIAS AUDIT REPORT")
            print("="*60)
            print(f" Generated: {report.get('execution_time', 'Unknown')}")
            print(f" Languages: {', '.join(report.get('languages_analyzed', []))}")
            print(f" Average SDI: {report.get('key_metrics', {}).get('average_sdi', 'N/A')}")
            print(f" Interpretation: {report.get('key_metrics', {}).get('sdi_interpretation', 'N/A')}")
            print(f" Total Flags: {report.get('key_metrics', {}).get('total_flags', 0)}")
            
            if report.get('cultural_preservation'):
                print(f"\n Cultural Items to Preserve: {len(report['cultural_preservation'])}")
                for item in report['cultural_preservation'][:5]:
                    print(f"   • {item[:70]}...")
        else:
            print("   No reports found. Run the pipeline first with --full")
    
    print("\n" + "="*70)
    print(f" Execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()