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

**Multilingual Bias Detection for Maternal Health AI in East Africa**
<img width="1536" height="1024" alt="banner" src="https://github.com/user-attachments/assets/5b510872-86fb-4b9b-aa0f-a3426839a5e6" />

**Overview**
MaHealthBiasAudit v2 is the first comprehensive AI auditing framework designed to detect, quantify, and explain language bias in maternal health question-answering systems across four East African languages: English, Luganda, Runyankore, and Swahili. This toolkit addresses a critical gap in AI fairness evaluation for low-resource African languages in healthcare contexts.

**Dataset**
https://dataverse.harvard.edu/file.xhtml?fileId=13738374

**Why This Matters**
Over 150 million people across East Africa speak Bantu languages, yet most health AI systems are trained primarily on English data. This creates significant disparities in:
1)Health information quality across languages
2)Cultural appropriateness of responses
3)Trustworthiness of AI-driven health advice
4)Patient outcomes in multilingual communities

**What Makes This Unique**
1. Linguistically-Informed Metrics
Unlike generic bias tools, our framework accounts for Bantu language morphology, including noun classes, prefix-suffix structures, and agglutinative patterns that cause 2-3x tokenization overhead.

2. Cultural Trust Scoring
We don't just measure translation accuracy—we evaluate whether health information preserves cultural practices, traditional beliefs, and locally-relevant concepts essential for maternal care.

3. Explainable Root Causes
Instead of opaque bias scores, we identify specific failure points: tokenization fertility penalties, out-of-vocabulary rates for health terms, and morphological misalignment.

**Quick Start**
------------------------------------------------------
1. git clone https://github.com/ronaldkato/MaHealthBiasAudit
2. cd MaHealthBiasAudit
3. pip install -r requirements.txt
4. python run_bias_audit.py --full --output-dir results

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
