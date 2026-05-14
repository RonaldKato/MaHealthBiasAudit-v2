"""
Configuration file for MaHealthBiasAudit v2
"""

# Languages in the study (4 core + 2 extended for Bantu analysis)
LANGUAGES = {
    'English': {'code': 'en', 'family': 'Indo-European', 'resource_level': 'high', 'script': 'Latin'},
    'Swahili': {'code': 'sw', 'family': 'Bantu', 'resource_level': 'medium', 'script': 'Latin', 'noun_classes': 18},
    'Yoruba': {'code': 'yo', 'family': 'Niger-Congo', 'resource_level': 'low', 'script': 'Latin'},
    'Amharic': {'code': 'am', 'family': 'Semitic', 'resource_level': 'low', 'script': 'Ethiopic'},
    'Luganda': {'code': 'lg', 'family': 'Bantu', 'resource_level': 'low', 'script': 'Latin', 'noun_classes': 15},
    'Runyankore': {'code': 'rn', 'family': 'Bantu', 'resource_level': 'very_low', 'script': 'Latin', 'noun_classes': 16}
}

# Primary languages for detailed analysis (focus of the study)
PRIMARY_LANGUAGES = ['English', 'Luganda', 'Runyankore', 'Swahili']

# Maternal health topics (based on WHO categories)
MATERNAL_TOPICS = {
    'antenatal_care': {
        'keywords': ['pregnant', 'pregnancy', 'folic', 'iron', 'calcium', 'nutrients', 'fetus', 'baby', 'prenatal'],
        'description': 'Care during pregnancy'
    },
    'labor_delivery': {
        'keywords': ['labor', 'labour', 'contractions', 'delivery', 'birth', 'cervical', 'water breaking', 'dilation'],
        'description': 'Childbirth and delivery'
    },
    'postnatal_care': {
        'keywords': ['postpartum', 'breastfeeding', 'new mother', 'recovery', 'after birth', 'postnatal'],
        'description': 'Care after delivery'
    },
    'mental_health': {
        'keywords': ['depression', 'mental health', 'counseling', 'support', 'anxiety', 'postpartum depression'],
        'description': 'Maternal mental health'
    },
    'child_health': {
        'keywords': ['vaccinations', 'vaccines', 'child', 'infant', 'BCG', 'Polio', 'Measles', 'immunization'],
        'description': 'Newborn and child health'
    },
    'emergency_referral': {
        'keywords': ['emergency', 'danger signs', 'referral', 'bleeding', 'convulsions', 'fever', 'severe'],
        'description': 'Emergency obstetric care'
    },
    'nutrition': {
        'keywords': ['nutrients', 'diet', 'eat', 'food', 'nutrition', 'vitamins', 'supplements', 'malnutrition'],
        'description': 'Maternal nutrition'
    },
    'family_planning': {
        'keywords': ['contraception', 'birth control', 'family planning', 'spacing', 'pregnancy prevention'],
        'description': 'Family planning services'
    }
}

# Bias thresholds (based on research proposal)
THRESHOLDS = {
    'sdi_high': 0.4,
    'sdi_moderate': 0.2,
    'tokenisation_parity': 1.5,
    'oov_rate': 0.15,
    'mas_threshold': 0.6,
    'jsd_high': 0.5,
    'f1_disparity_high': 0.3,
    'language_purity_high': 0.7,
    'topic_purity_target': 0.8
}

# Embedding models (from proposal)
EMBEDDING_MODELS = {
    'LaBSE': 'sentence-transformers/LaBSE',
    'multilingual-e5': 'intfloat/multilingual-e5-large',
    'LASER3': 'laser3'
}

# Model configurations for fine-tuning (from proposal Table 2)
MODEL_CONFIGS = {
    'mBERT': 'bert-base-multilingual-cased',
    'XLM-R': 'xlm-roberta-base',
    'SERENGETI': 'Davlan/afro-xlmr-base',  # Proxy for SERENGETI
    'AfriBERTa': 'castorini/afriberta_base'
}

# Fine-tuning conditions (5x3 matrix from proposal)
FINE_TUNE_CONDITIONS = ['FT-EN', 'FT-SW', 'FT-YO', 'FT-AM', 'FT-MULTI']

# Interrogative structure patterns (from Figure 4 in proposal)
INTERROGATIVE_PATTERNS = {
    'English': {'pattern': r'^(what|when|where|why|how|which|who)', 'type': 'wh_fronted'},
    'Swahili': {'pattern': r'(je|nini|gani|lini|wapi|kwa nini|vipi)', 'type': 'verb_internal'},
    'Luganda': {'pattern': r'(ki|nya|li|wa)', 'type': 'verb_suffix', 'note': 'Interrogative suffix -ki attached to verb'},
    'Runyankore': {'pattern': r'(ki|nya)', 'type': 'sentence_final', 'note': 'Interrogative at sentence-final position'},
    'Yoruba': {'pattern': r'(kí|níbo|nígbà|kí ló|báwo|ta)', 'type': 'wh_fronted'},
    'Amharic': {'pattern': r'(ምን|የት|መቼ|ለምን|እንዴት|ማን)', 'type': 'wh_fronted'}
}

# Cultural terminology database (from Trust-Aware Module)
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

# Dialect markers for variance detection
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

# Visualization settings
VIZ_SETTINGS = {
    'color_palette': 'viridis',
    'figure_dpi': 150,
    'save_figures': True,
    'interactive': True
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Output directories
OUTPUT_DIR = "mahealth_bias_output"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"
MODELS_DIR = f"{OUTPUT_DIR}/models"