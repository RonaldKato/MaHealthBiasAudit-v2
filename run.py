#!/usr/bin/env python3
"""
MaHealthBiasAudit - Runner Script
Execute the bias audit pipeline
"""

import sys
import os
import argparse
from datetime import datetime

# CRITICAL: Set environment variables BEFORE any other imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import MaHealthBiasAudit
from config import OUTPUT_DIR, REPORTS_DIR, FIGURES_DIR, PRIMARY_LANGUAGES
from utils import setup_logger
import logging
import webbrowser
import pandas as pd
import numpy as np


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MaHealthBiasAudit - Maternal Health Bias Detection')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-embeddings', action='store_true',
                       help='Skip embedding computation (saves memory)')
    parser.add_argument('--open-viz', action='store_true',
                       help='Open visualizations in browser after generation')
    parser.add_argument('--show-viz', action='store_true',
                       help='Display all visualizations on screen after generation')
    return parser.parse_args()


def print_banner():
    """Print welcome banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║     MaHealthBiasAudit - Maternal Health Bias Detection           ║
    ║                                                                  ║
    ║     A comprehensive bias audit framework for                     ║
    ║     multilingual maternal health datasets                        ║
    ║                                                                  ║
    ║     Languages: English, Swahili, Luganda, Runyankore             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_summary_table(report):
    """Print formatted summary table"""
    print("\n" + "=" * 70)
    print("BIAS AUDIT SUMMARY")
    print("=" * 70)
    
    print(f"\n  Dataset Statistics:")
    print(f"    ├─ Languages: {', '.join(report['languages'])}")
    print(f"    ├─ Total Questions: {report['total_questions']}")
    print(f"    └─ Total Answers: {report['total_answers']}")
    
    print(f"\n  Key Metrics:")
    print(f"    ├─ Average SDI: {report['key_metrics']['average_sdi']:.4f}")
    print(f"    ├─ Bias Level: {report['key_metrics']['bias_level']}")
    print(f"    └─ Total Flags: {report['key_metrics']['total_flags']}")
    
    print(f"\n  SDI Ranking (vs English):")
    for lang, sdi in sorted(report['sdi_ranking'].items(), key=lambda x: x[1], reverse=True):
        print(f"    ├─ {lang}: {sdi:.4f}")
    
    print(f"\n  Root Cause Distribution:")
    for cause, count in report['rca_distribution'].items():
        print(f"    ├─ {cause}: {count}")
    
    print(f"\n  Recommendations:")
    for rec in report.get('recommendations', [])[:5]:
        print(f"    ├─ {rec}")
    
    print(f"\n  Output Locations:")
    print(f"    ├─ Reports: {REPORTS_DIR}")
    print(f"    ├─ Figures (PNG): {FIGURES_DIR}")
    print(f"    └─ Figures (HTML): {FIGURES_DIR}")
    
    print("\n" + "=" * 70)


def list_generated_visualizations():
    """List all generated visualization files"""
    print("\n" + "=" * 70)
    print("GENERATED VISUALIZATIONS")
    print("=" * 70)
    
    viz_files = []
    if os.path.exists(FIGURES_DIR):
        for f in sorted(os.listdir(FIGURES_DIR)):
            if f.endswith(('.png', '.html')):
                viz_files.append(f)
    
    if viz_files:
        print("\n  The following visualization files were generated:\n")
        for i, f in enumerate(viz_files, 1):
            print(f"    {i:2d}. {f}")
        print(f"\n  Total: {len(viz_files)} files")
        print(f"  Location: {FIGURES_DIR}")
    else:
        print("\n  No visualization files found.")
    
    print("\n" + "=" * 70)


def open_visualizations():
    """Open generated HTML visualizations in browser"""
    if not os.path.exists(FIGURES_DIR):
        print("No visualizations found to open.")
        return
    
    html_files = [f for f in os.listdir(FIGURES_DIR) if f.endswith('.html')]
    
    if html_files:
        print(f"\nOpening {len(html_files)} visualizations in browser...")
        for html_file in sorted(html_files):
            filepath = os.path.join(FIGURES_DIR, html_file)
            webbrowser.open(f'file://{filepath}')
        print("Browser tabs opened for each HTML visualization.")
    else:
        print("No HTML visualization files found to open.")


def main():
    """Main execution function"""
    args = parse_arguments()
    
    print_banner()
    
    # Setup logger
    logger = setup_logger('runner', level=logging.DEBUG if args.verbose else logging.INFO)
    
    logger.info(f"Starting bias audit at {datetime.now()}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Initialize and run audit
        audit = MaHealthBiasAudit()
        report = audit.run()
        
        # Print summary
        print_summary_table(report)
        
        # Extract data from results for visualizations
        results = audit.results
        cross_lingual = results.get('cross_lingual', {})
        statistical = results.get('statistical', {})
        preprocessing = results.get('preprocessing', {})
        linguistic = results.get('linguistic', {})
        
        # Prepare data for visualizations
        sdi_matrix = cross_lingual.get('sdi_matrix', pd.DataFrame())
        length_stats = statistical.get('response_length_stats', pd.DataFrame())
        tp_df = preprocessing.get('tokenisation_parity', pd.DataFrame())
        trust_results = linguistic.get('trust_aware_results', [])
        rca_counts = cross_lingual.get('error_categories', {})
        sdi_ranking = report.get('sdi_ranking', {})
        flags = report.get('flags', [])
        
        # Prepare length_data for violin plot
        answers_by_lang = preprocessing.get('normalised_texts', {})
        length_data = {}
        for lang in PRIMARY_LANGUAGES:
            if lang in answers_by_lang:
                lengths = [len(text.split()) for text in answers_by_lang[lang]]
                length_data[lang] = lengths
        
        # Get test results from statistical audit
        test_results = statistical.get('statistical_tests', [])
        
        # Get category stats from preprocessing
        category_stats = None
        if 'metadata' in preprocessing and 'category_metadata' in preprocessing['metadata']:
            cat_data = []
            for cat, meta in preprocessing['metadata']['category_metadata'].items():
                for lang, count in meta['num_answers_per_lang'].items():
                    cat_data.append({
                        'Category': cat,
                        'Language': lang,
                        'Avg_Length': count  # Placeholder - would need actual lengths
                    })
            if cat_data:
                category_stats = pd.DataFrame(cat_data)
        
        # Get dataset statistics
        stats_df = preprocessing.get('statistics', pd.DataFrame())
        
        # Get recommendations
        recommendations = report.get('recommendations', [])
        
        # Get outliers
        outliers = statistical.get('outliers', [])
        
        # Get n-gram data (from statistical audit)
        ngram_data = {}
        bigram_stats = statistical.get('bigram_analysis', pd.DataFrame())
        if not bigram_stats.empty:
            for _, row in bigram_stats.iterrows():
                lang = row['Language']
                top_ngrams = row.get('Top_2-grams', [])
                if isinstance(top_ngrams, list) and top_ngrams:
                    ngram_data[lang] = [(ng[0] if isinstance(ng, tuple) else ng, ng[1] if isinstance(ng, tuple) and len(ng) > 1 else 1) for ng in top_ngrams[:10]]
        
        # Get vocabulary stats
        vocab_stats = statistical.get('vocabulary_richness', pd.DataFrame())
        
        # Call save_all_visualizations with all collected data
        print("\n" + "=" * 70)
        print("SAVING ALL 18 VISUALIZATIONS")
        print("=" * 70)
        print(f"Output directory: {FIGURES_DIR}\n")
        
        audit.viz.save_all_visualizations(
            sdi_matrix=sdi_matrix,
            length_stats=length_stats,
            tp_df=tp_df,
            trust_results=trust_results,
            rca_counts=rca_counts,
            sdi_ranking=sdi_ranking,
            flags=flags,
            length_data=length_data,
            avg_sdi=report['key_metrics']['average_sdi'],
            bias_level=report['key_metrics']['bias_level'],
            total_flags=report['key_metrics']['total_flags'],
            test_results=test_results,
            category_stats=category_stats,
            stats_df=stats_df,
            recommendations=recommendations,
            outliers=outliers,
            ngram_data=ngram_data,
            report=report,
            vocab_stats=vocab_stats
        )
        
        # List generated visualizations
        list_generated_visualizations()
        
        # Open visualizations in browser if requested
        if args.open_viz:
            open_visualizations()
        
        # Display all visualizations on screen if requested
        if args.show_viz:
            print("\n" + "=" * 70)
            print("DISPLAYING VISUALIZATIONS ON SCREEN")
            print("=" * 70)
            audit.viz.show()
        
        logger.info("Bias audit completed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during bias audit: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())