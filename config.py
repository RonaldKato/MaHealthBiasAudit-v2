"""
Configuration file for MaHealthBiasAudit v2
Centralized configuration for the entire bias audit pipeline
"""

import os
from datetime import datetime

# ============================================================================
# EXPERIMENT METADATA
# ============================================================================

EXPERIMENT_NAME = "MaHealthBiasAudit_v2"
EXPERIMENT_VERSION = "2.0.0"
EXECUTION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# LANGUAGES IN THE STUDY
# ============================================================================

LANGUAGES = {
    'English': {
        'code': 'en', 
        'family': 'Indo-European', 
        'resource_level': 'high', 
        'script': 'Latin',
        'morphological_complexity': 1.0,
        'has_tones': False
    },
    'Swahili': {
        'code': 'sw', 
        'family': 'Bantu', 
        'resource_level': 'medium', 
        'script': 'Latin',
        'morphological_complexity': 1.6,
        'has_tones': False,
        'noun_classes': 18
    },
    'Yoruba': {
        'code': 'yo', 
        'family': 'Niger-Congo', 
        'resource_level': 'low', 
        'script': 'Latin',
        'morphological_complexity': 1.8,
        'has_tones': True
    },
    'Amharic': {
        'code': 'am', 
        'family': 'Semitic', 
        'resource_level': 'low', 
        'script': 'Ethiopic',
        'morphological_complexity': 2.0,
        'has_tones': False
    },
    'Luganda': {
        'code': 'lg', 
        'family': 'Bantu', 
        'resource_level': 'low', 
        'script': 'Latin',
        'morphological_complexity': 2.2,
        'has_tones': True,
        'noun_classes': 15
    },
    'Runyankore': {
        'code': 'rn', 
        'family': 'Bantu', 
        'resource_level': 'very_low', 
        'script': 'Latin',
        'morphological_complexity': 2.5,
        'has_tones': True,
        'noun_classes': 16
    }
}

# Primary languages for detailed analysis
PRIMARY_LANGUAGES = ['English', 'Swahili', 'Yoruba', 'Amharic']

# Extended Bantu languages for comparative analysis
BANTU_LANGUAGES = ['Swahili', 'Luganda', 'Runyankore']

# ============================================================================
# MATERNAL HEALTH TOPICS (WHO Categories)
# ============================================================================

MATERNAL_TOPICS = {
    'antenatal_care': {
        'keywords': ['pregnant', 'pregnancy', 'folic', 'iron', 'calcium', 'nutrients', 
                    'fetus', 'baby', 'prenatal', 'antenatal', 'vitamins'],
        'description': 'Care during pregnancy',
        'color': '#3498db'
    },
    'labor_delivery': {
        'keywords': ['labor', 'labour', 'contractions', 'delivery', 'birth', 'cervical', 
                    'water breaking', 'dilation', 'childbirth'],
        'description': 'Childbirth and delivery',
        'color': '#e74c3c'
    },
    'postnatal_care': {
        'keywords': ['postpartum', 'breastfeeding', 'new mother', 'recovery', 'after birth', 
                    'postnatal', 'newborn'],
        'description': 'Care after delivery',
        'color': '#2ecc71'
    },
    'mental_health': {
        'keywords': ['depression', 'mental health', 'counseling', 'support', 'anxiety', 
                    'postpartum depression', 'psychological'],
        'description': 'Maternal mental health',
        'color': '#f39c12'
    },
    'child_health': {
        'keywords': ['vaccinations', 'vaccines', 'child', 'infant', 'BCG', 'Polio', 
                    'Measles', 'immunization', 'newborn health'],
        'description': 'Newborn and child health',
        'color': '#9b59b6'
    },
    'emergency_referral': {
        'keywords': ['emergency', 'danger signs', 'referral', 'bleeding', 'convulsions', 
                    'fever', 'severe', 'urgent'],
        'description': 'Emergency obstetric care',
        'color': '#e67e22'
    },
    'nutrition': {
        'keywords': ['nutrients', 'diet', 'eat', 'food', 'nutrition', 'vitamins', 
                    'supplements', 'malnutrition'],
        'description': 'Maternal nutrition',
        'color': '#1abc9c'
    },
    'family_planning': {
        'keywords': ['contraception', 'birth control', 'family planning', 'spacing', 
                    'pregnancy prevention'],
        'description': 'Family planning services',
        'color': '#34495e'
    }
}

# ============================================================================
# BIAS THRESHOLDS (Based on research proposal)
# ============================================================================

THRESHOLDS = {
    # Semantic Divergence
    'sdi_high': 0.4,
    'sdi_moderate': 0.2,
    
    # Tokenisation
    'tokenisation_parity': 1.5,
    'oov_rate': 0.15,
    
    # Morphological
    'mas_threshold': 0.6,
    
    # Distributional
    'jsd_high': 0.5,
    
    # Performance
    'f1_disparity_high': 0.3,
    'exact_match_threshold': 0.7,
    
    # Clustering
    'language_purity_high': 0.7,
    'topic_purity_target': 0.8,
    
    # Trust
    'trust_score_target': 0.7
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

EMBEDDING_MODELS = {
    'LaBSE': 'sentence-transformers/LaBSE',
    'multilingual-e5': 'intfloat/multilingual-e5-large',
    'LASER3': 'laser3'
}

MODEL_CONFIGS = {
    'mBERT': {
        'name': 'bert-base-multilingual-cased',
        'type': 'encoder-only',
        'max_length': 512,
        'fertility_baseline': 1.0
    },
    'XLM-R': {
        'name': 'xlm-roberta-base',
        'type': 'encoder-only',
        'max_length': 512,
        'fertility_baseline': 0.9
    },
    'AfriBERTa': {
        'name': 'castorini/afriberta_base',
        'type': 'encoder-only',
        'max_length': 512,
        'fertility_baseline': 0.8
    },
    'SERENGETI': {
        'name': 'Davlan/afro-xlmr-base',
        'type': 'encoder-only',
        'max_length': 512,
        'fertility_baseline': 0.85
    }
}

# Fine-tuning conditions (5x3 matrix)
FINE_TUNE_CONDITIONS = ['FT-EN', 'FT-SW', 'FT-YO', 'FT-AM', 'FT-MULTI']

# ============================================================================
# LINGUISTIC PATTERNS
# ============================================================================

INTERROGATIVE_PATTERNS = {
    'English': {'pattern': r'^(what|when|where|why|how|which|who)', 'type': 'wh_fronted'},
    'Swahili': {'pattern': r'(je|nini|gani|lini|wapi|kwa nini|vipi)', 'type': 'verb_internal'},
    'Luganda': {'pattern': r'(ki|nya|li|wa)', 'type': 'verb_suffix', 'note': 'Interrogative suffix -ki attached to verb'},
    'Runyankore': {'pattern': r'(ki|nya)', 'type': 'sentence_final', 'note': 'Interrogative at sentence-final position'},
    'Yoruba': {'pattern': r'(kí|níbo|nígbà|kí ló|báwo|ta)', 'type': 'wh_fronted'},
    'Amharic': {'pattern': r'(ምን|የት|መቼ|ለምን|እንዴት|ማን)', 'type': 'wh_fronted'}
}

# ============================================================================
# CULTURAL TERMINOLOGY (Trust-Aware Module)
# ============================================================================

CULTURAL_TERMINOLOGY = {
    'Luganda': {
        'okuzaala': ('childbirth', 'to give birth', 0.95, True),
        'omusawo gw\'ebisaayiro': ('traditional healer', 'herbalist', 0.85, False),
        'eddagala ly\'ebisigire': ('herbal medicine', 'traditional medicine', 0.90, True),
        'okulongoosa': ('postnatal care', 'traditional postpartum care', 0.88, True),
        'essanyu ly\'omwana': ('child wellness', 'baby\'s joy/health', 0.80, False)
    },
    'Runyankore': {
        'okuzaara': ('childbirth', 'to give birth', 0.95, True),
        'omugabe w\'ebijukano': ('traditional healer', 'herbal specialist', 0.85, False),
        'ebibazi': ('herbal medicine', 'traditional remedies', 0.90, True),
        'okwireeza': ('postnatal period', 'recovery period', 0.88, True),
        'ekihango': ('ceremonial meal', 'postpartum celebration', 0.82, False)
    },
    'Swahili': {
        'majaliwa': ('blessed pregnancy', 'fortunate conception', 0.90, False),
        'ugonjwa wa kusema': ('psychosomatic condition', 'illness from speech', 0.70, False),
        'dawa za asili': ('herbal medicine', 'traditional medicine', 0.85, True),
        'mganga': ('traditional healer', 'herbalist', 0.80, False),
        'kitanda cha uzazi': ('birthing bed', 'traditional delivery bed', 0.88, True)
    }
}

# ============================================================================
# DIALECT MARKERS
# ============================================================================

DIALECT_MARKERS = {
    'Swahili': {
        'Coastal': ['sana', 'kabisa', 'kweli', 'hasa'],
        'Standard': ['vizuri', 'safi', 'kamilifu', 'sahihi'],
        'Ugandan': ['mzuri', 'bora', 'vizuri sana', 'sawa']
    },
    'Luganda': {
        'Central': ['buli', 'nga', 'nnyo', 'ddene'],
        'Rural': ['bwebwe', 'nti', 'nyo', 'nene'],
        'Urban': ['baazi', 'ssebo', 'nyabo', 'kikulu']
    },
    'Runyankore': {
        'Standard': ['nka', 'kandi', 'nibwe', 'omuntu'],
        'Rural': ['eka', 'nibwo', 'omundu', 'bukuru']
    }
}

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

OUTPUT_DIR = "mahealth_bias_output"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"
MODELS_DIR = f"{OUTPUT_DIR}/models"
LOGS_DIR = f"{OUTPUT_DIR}/logs"

# Create directories
for dir_path in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

VIZ_SETTINGS = {
    'color_palette': 'viridis',
    'figure_dpi': 150,
    'save_figures': True,
    'interactive': True,
    'font_family': 'sans-serif',
    'font_size': 12
}

# ============================================================================
# RANDOM SEED FOR REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

# ============================================================================
# API KEYS (Optional)
# ============================================================================

# For Gemini API integration (optional)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', None)