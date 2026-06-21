# MaHealthBiasAudit v2

**Multilingual Bias Detection for Maternal Health AI in East Africa**

## Overview

MaHealthBiasAudit v2 is the first comprehensive AI auditing framework designed to detect, quantify, and explain language bias in maternal health question-answering systems across four East African languages: English, Luganda, Runyankore, and Swahili.

## Pipeline Stages

1. **INPUT**: Multilingual Maternal Health Dataset (4 languages, 8 categories, ~4,500 answers)
2. **PREPROCESSING**: Normalisation → Tokenisation
3. **MULTI-LAYER BIAS DETECTION**:
   - Stratum I: Rule-Based Detection (Lexicons, Heuristics, Linguistic Patterns)
   - Stratum II: Classifier-Based Detection (mBERT, XLM-R, RoBERTa)
   - Stratum III: Python-Based Detection (Pandas, Scikit-learn, Sentence-Transformers)
4. **CROSS-LINGUAL ANALYSIS**: Bias Comparison, Cultural Adaptation, Consistency Mapping
5. **OUTPUT**: Bias Detection Reports, High-Risk Flagging, Recommendations

## Quick Start

```bash
# Clone the repository
git clone https://github.com/RonaldKato/MaHealthBiasAudit-v2

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python run.py --full

# Show bias characteristics only
python run.py --show-bias-characteristics

# Skip experiments
python run.py --full --no-experiments

# Skip validation
python run.py --full --no-validation