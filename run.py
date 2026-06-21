#!/usr/bin/env python3
"""
MaHealthBiasAudit - Runner Script
Execute the complete bias audit pipeline with experiments and validation
"""

import sys
import os
import argparse

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from main import MaHealthBiasAudit
from utils import print_bias_characteristics


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MaHealthBiasAudit - Maternal Health Bias Detection')
    parser.add_argument('--full', action='store_true', default=True,
                       help='Run full audit pipeline')
    parser.add_argument('--no-experiments', action='store_true',
                       help='Skip experiments with increasing dataset sizes')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip validation on MOTHER dataset')
    parser.add_argument('--show-bias-characteristics', action='store_true',
                       help='Display bias-prone sentence characteristics')
    parser.add_argument('--output-dir', type=str, default=config.OUTPUT_DIR,
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=config.RANDOM_SEED,
                       help='Random seed for reproducibility')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip generating visualizations')
    parser.add_argument('--skip-samples', action='store_true',
                       help='Skip sample extraction')
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    
    if args.show_bias_characteristics:
        print_bias_characteristics()
        return 0
    
    print("\n" + "="*70)
    print("MAHEALTHBIASAUDIT - MATERNAL HEALTH BIAS DETECTION")
    print("="*70)
    print(f"\n   Output Directory: {args.output_dir}")
    print(f"   Experiments: {'DISABLED' if args.no_experiments else 'ENABLED'}")
    print(f"   Validation: {'DISABLED' if args.no_validation else 'ENABLED'}")
    print(f"   Visualizations: {'DISABLED' if args.skip_viz else 'ENABLED'}")
    print(f"   Sample Extraction: {'DISABLED' if args.skip_samples else 'ENABLED'}")
    print(f"   Random Seed: {args.seed}")
    print(f"   Languages: {config.PRIMARY_LANGUAGES}")
    print(f"   Experiment Sizes: {config.EXPERIMENT_SIZES}")
    
    # Update config output directory
    if args.output_dir != config.OUTPUT_DIR:
        config.OUTPUT_DIR = args.output_dir
        config.FIGURES_DIR = f"{args.output_dir}/figures"
        config.REPORTS_DIR = f"{args.output_dir}/reports"
        config.VALIDATION_DIR = f"{args.output_dir}/validation"
        config.EXPERIMENTS_DIR = f"{args.output_dir}/experiments"
        config.SAMPLES_DIR = f"{args.output_dir}/samples"
        
        for directory in [config.OUTPUT_DIR, config.FIGURES_DIR, config.REPORTS_DIR, 
                         config.LOGS_DIR, config.VALIDATION_DIR, config.EXPERIMENTS_DIR,
                         config.SAMPLES_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    config.RANDOM_SEED = args.seed
    
    try:
        audit = MaHealthBiasAudit()
        main_report = audit.run_full_audit(
            run_experiments=not args.no_experiments,
            run_validation=not args.no_validation
        )
        

        
        # Print sample summary
        if audit.sample_results:
            print(f"\n  Sample Summary:")
            for lang in config.PRIMARY_LANGUAGES:
                if lang in audit.sample_results.get('unbiased', {}):
                    unbiased = len(audit.sample_results['unbiased'].get(lang, []))
                    biased = len(audit.sample_results['biased'].get(lang, []))
                    print(f"   - {lang}: {unbiased} unbiased, {biased} biased samples")
            print(f"  Sample report: {config.SAMPLES_DIR}")
        
        # ============================================================
        # MOTHER DATASET VALIDATION DETAILS - ADDED SECTION
        # ============================================================
        if not args.no_validation:
            print("\n" + "="*70)
            print(" MOTHER DATASET VALIDATION DETAILS")
            print("="*70)
            
            validation_results = audit.validation_results
            if validation_results and validation_results.get('status') == 'PASSED':
                metrics = validation_results.get('metrics', {})
                lang_sdi = validation_results.get('language_sdi', {})
                
                print(f"\n  Overall SDI: {metrics.get('sdi_percentage', 'N/A')}")
                print(f"  Languages: {validation_results.get('languages', [])}")
                print(f"  Bias Level: {metrics.get('full_bias_level', 'N/A')}")
                print(f"  Total Flags: {metrics.get('full_flags', 0)} (Critical: {metrics.get('critical_flags', 0)})")
                
                print(f"\n  Language-specific SDI (vs English):")
                for lang, sdi in lang_sdi.items():
                    print(f"    • {lang}: {sdi:.1f}%")
                
                # Print statistical comparisons with Benjamini-Hochberg adjustment
                stat_comps = validation_results.get('statistical_comparisons', [])
                if stat_comps:
                    print(f"\n  Statistical Comparisons (Benjamini-Hochberg adjusted):")
                    print(f"  {'Comparison':<30} {'p-value':<12} {'Significant':<12}")
                    print(f"  {'-'*54}")
                    for comp in stat_comps:
                        comparison = comp.get('comparison', 'Unknown')[:28]
                        p_val = comp.get('p_value_adjusted', 1.0)
                        status = "✓ Yes" if comp.get('significant_adjusted') else "✗ No"
                        print(f"  {comparison:<30} {p_val:<12.4f} {status:<12}")
                
                # Flag distribution
                flag_summary = validation_results.get('flag_summary', {})
                if flag_summary:
                    print(f"\n  Flag Distribution:")
                    print(f"    Total: {flag_summary.get('total_flags', 0)}")
                    print(f"    Critical: {flag_summary.get('critical_flags', 0)}")
                    
                    by_type = flag_summary.get('by_type', {})
                    if by_type:
                        print(f"    By Type:")
                        for flag_type, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True):
                            print(f"      • {flag_type}: {count}")
                    
                    by_language = flag_summary.get('by_language', {})
                    if by_language:
                        print(f"    By Language:")
                        for lang, count in sorted(by_language.items(), key=lambda x: x[1], reverse=True):
                            print(f"      • {lang}: {count}")
                
                # Print validation message
                print(f"\n {validation_results.get('message', 'Validation completed successfully')}")
                
            elif validation_results and validation_results.get('status') == 'FAILED':
                print(f"\n Validation Failed: {validation_results.get('error', 'Unknown error')}")
            else:
                print(f"\n  ⚠ No validation results available")
        
        return 0
        
    except Exception as e:
        print(f"\n Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())