"""
Configuration file for MaHealthBiasAudit v2
languages: English, Swahili, Luganda, Runyankore
"""

import os
from datetime import datetime

# ============================================================================
# EXPERIMENT METADATA
# ============================================================================

EXPERIMENT_NAME = "MaHealthBiasAudit"
EXPERIMENT_VERSION = "1.0.0"
EXECUTION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# LANGUAGES
# ============================================================================

PRIMARY_LANGUAGES = ['English', 'Swahili', 'Luganda', 'Runyankore']

LANGUAGES = {
    'English': {
        'code': 'en', 
        'family': 'Indo-European', 
        'resource_level': 'high', 
        'script': 'Latin',
        'morphological_complexity': 1.0,
        'has_tones': False,
        'word_order': 'SVO'
    },
    'Swahili': {
        'code': 'sw', 
        'family': 'Bantu', 
        'resource_level': 'medium', 
        'script': 'Latin',
        'morphological_complexity': 1.6,
        'has_tones': False,
        'noun_classes': 18,
        'word_order': 'SVO'
    },
    'Luganda': {
        'code': 'lg', 
        'family': 'Bantu', 
        'resource_level': 'low', 
        'script': 'Latin',
        'morphological_complexity': 2.2,
        'has_tones': True,
        'noun_classes': 15,
        'word_order': 'SVO'
    },
    'Runyankore': {
        'code': 'rn', 
        'family': 'Bantu', 
        'resource_level': 'very_low', 
        'script': 'Latin',
        'morphological_complexity': 2.5,
        'has_tones': True,
        'noun_classes': 16,
        'word_order': 'SVO'
    }
}

# ============================================================================
# YOUR 8 CATEGORIES (Based on dataset)
# ============================================================================

DATASET_CATEGORIES = [
    "about_healthcare_access",
    "about_medical_history_lifestyle",
    "about_mental_health_support",
    "about_recovery_baby_care",
    "about_recovery_health",
    "about_symptoms_concerns",
    "about_traditional_cultural_practices",
    "community_cultural_considerations"
]

CATEGORY_TOPICS = {
    "about_healthcare_access": "Healthcare Access",
    "about_medical_history_lifestyle": "Medical History & Lifestyle",
    "about_mental_health_support": "Mental Health & Support",
    "about_recovery_baby_care": "Recovery & Baby Care",
    "about_recovery_health": "Recovery & Health",
    "about_symptoms_concerns": "Symptoms & Concerns",
    "about_traditional_cultural_practices": "Traditional & Cultural Practices",
    "community_cultural_considerations": "Community & Cultural Considerations"
}

# ============================================================================
# BIAS THRESHOLDS
# ============================================================================

THRESHOLDS = {
    'sdi_high': 0.4,
    'sdi_moderate': 0.2,
    'tokenisation_parity': 1.5,
    'oov_rate': 0.15,
    'mas_threshold': 0.6,
    'jsd_high': 0.5,
    'f1_disparity_high': 0.3,
    'trust_score_target': 0.7
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    'mBERT': {'name': 'bert-base-multilingual-cased', 'fertility_baseline': 1.0},
    'XLM-R': {'name': 'xlm-roberta-base', 'fertility_baseline': 0.9},
    'AfriBERTa': {'name': 'castorini/afriberta_base', 'fertility_baseline': 0.8}
}

# ============================================================================
# INTERROGATIVE PATTERNS 
# ============================================================================

INTERROGATIVE_PATTERNS = {
    'English': {'pattern': r'^(what|when|where|why|how|which|who|have|do|are|is|can|will)', 'type': 'wh_fronted'},
    'Swahili': {'pattern': r'(je|nini|gani|lini|wapi|kwa nini|vipi|je|mbona)', 'type': 'mixed'},
    'Luganda': {'pattern': r'(ki|nya|li|wa|ana|ali|kati)', 'type': 'verb_suffix'},
    'Runyankore': {'pattern': r'(ki|nya|aha|obu|eki|nki)', 'type': 'sentence_final'}
}

# ============================================================================
# CULTURAL TERMINOLOGY 
# ============================================================================

CULTURAL_TERMINOLOGY = {
    'Luganda': {
        'yee': ('affirmation', 'yes', 0.70, False),
        'omwana': ('child', 'baby', 0.85, True),
        'namufiirwa': ('loss', 'lost the baby', 0.90, True),
        'nsobola': ('ability', 'I can', 0.70, False),
        'obutimba bw\'ensiri': ('mosquito nets', 'malaria prevention', 0.85, True),
        'matooke': ('food', 'cooking bananas', 0.75, False),
        'akawunga': ('food', 'maize flour/posho', 0.70, False),
        'okusaba': ('practice', 'praying', 0.85, False),
        'Katonda': ('deity', 'God', 0.80, False),
        'omwami': ('family', 'husband', 0.75, False),
        'eddagala': ('medicine', 'herbal remedy', 0.85, True),
        'entangawuzzi': ('herb', 'ginger', 0.80, True)
    },
    'Runyankore': {
        'eego': ('affirmation', 'yes', 0.70, False),
        'omwana': ('child', 'baby', 0.85, True),
        'nkaferwa': ('loss', 'lost the baby', 0.90, True),
        'nyine': ('possession', 'I have', 0.70, False),
        'obutimba': ('nets', 'mosquito nets', 0.85, True),
        'ebitookye': ('food', 'cooking bananas', 0.75, False),
        'akahunga': ('food', 'maize flour', 0.70, False),
        'okushaba': ('practice', 'praying', 0.85, False),
        'Ruhanga': ('deity', 'God', 0.80, False),
        'omushaija': ('family', 'husband', 0.75, False),
        'ebibazi': ('medicine', 'herbal remedies', 0.85, True),
        'entangawuuzi': ('herb', 'ginger', 0.80, True)
    },
    'Swahili': {
        'ndiyo': ('affirmation', 'yes', 0.70, False),
        'mtoto': ('child', 'baby', 0.85, True),
        'nilipoteza': ('loss', 'lost the baby', 0.90, True),
        'ninaweza': ('ability', 'I can', 0.70, False),
        'vyandalua': ('nets', 'mosquito nets', 0.85, True),
        'matooke': ('food', 'cooking bananas', 0.75, False),
        'posho': ('food', 'maize flour', 0.70, False),
        'kusali': ('practice', 'praying', 0.85, False),
        'Mungu': ('deity', 'God', 0.80, False),
        'mume': ('family', 'husband', 0.75, False),
        'dawa za asili': ('medicine', 'traditional medicine', 0.85, True),
        'tangawizi': ('herb', 'ginger', 0.80, True)
    }
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_DIR = "mahealth_bias_output"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
FIGURES_ENHANCED_DIR = f"{OUTPUT_DIR}/figures_enhanced"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"

for dir_path in [OUTPUT_DIR, FIGURES_DIR, FIGURES_ENHANCED_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# COLORS FOR VISUALIZATION
# ============================================================================

LANG_COLORS = {
    'English': '#2E86AB',
    'Swahili': '#A23B72',
    'Luganda': '#F18F01',
    'Runyankore': '#C73E1D'
}

RCA_COLORS = {
    'TOKENISATION': '#3498DB',
    'QUERY_STRUCTURE': '#F39C12',
    'CULTURAL': '#27AE60',
    'MORPHOLOGY': '#9B59B6',
    'UNKNOWN': '#95A5A6'
}

RANDOM_SEED = 42