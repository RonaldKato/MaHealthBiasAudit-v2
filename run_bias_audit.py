#!/usr/bin/env python3
"""
MaHealthBiasAudit - Runner Script
Run the complete bias audit
"""

import argparse
import sys
import os
import random
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='MaHealthBiasAudit - Bias Detection for Maternal Health QA')
    parser.add_argument('--full', action='store_true', help='Run full audit pipeline')
    parser.add_argument('--output-dir', default='mahealth_bias_output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if not args.full:
        args.full = True
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Update config if needed
    import config
    if args.output_dir != 'mahealth_bias_output':
        config.OUTPUT_DIR = args.output_dir
        config.FIGURES_DIR = f"{args.output_dir}/figures"
        config.REPORTS_DIR = f"{args.output_dir}/reports"
    
    from main import MaHealthBiasAudit
    
    print("\n" + "="*70)
    print("MaHealthBiasAudit - Maternal Health Bias Detection")
    print(f"   Output: {config.OUTPUT_DIR}")
    print(f"   Seed: {args.seed}")
    print("="*70)
    
    audit = MaHealthBiasAudit()
    results = audit.run()
    
    print(f"\n Results saved to {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()