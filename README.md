**Multilingual Bias Detection for Maternal Health AI in East Africa**
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
1. git clone https://github.com/yourlab/MaHealthBiasAudit
2. cd MaHealthBiasAudit
3. pip install -r requirements.txt
4. python run_bias_audit.py --full --output-dir results
