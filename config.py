"""
MaHealthBiasAudit - Configuration Module
Centralized configuration for the entire bias audit system
"""

import os
from datetime import datetime

# ============================================================
# Basic Configuration
# ============================================================

# Languages in the dataset (from main.py)
PRIMARY_LANGUAGES = ['English', 'Swahili', 'Luganda', 'Runyankore']

# Dataset categories
DATA_CATEGORIES = [
    'about_healthcare_access',
    'about_medical_history_lifestyle',
    'about_mental_health_support',
    'about_recovery_baby_care',
    'about_recovery_health',
    'about_symptoms_concerns',
    'about_traditional_cultural_practices',
    'community_cultural_considerations'
]

# ============================================================
# Output Directories
# ============================================================

# Base output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')

# Create directories
for directory in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# Execution Configuration
# ============================================================

RANDOM_SEED = 42
EXECUTION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# Analysis Configuration
# ============================================================

# Statistical thresholds
SIGNIFICANCE_ALPHA = 0.05
MIN_ANSWER_LENGTH = 5
MAX_ANSWER_LENGTH = 5000

# Linguistic thresholds
MAX_TOKEN_FERTILITY = 1.5
SENTENCE_PIECING_THRESHOLD = 0.3
WORD_PIECING_THRESHOLD = 0.2

# Model configuration
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128

# Cross-lingual thresholds
SDI_THRESHOLD_HIGH = 0.4
SDI_THRESHOLD_MODERATE = 0.2
RCA_TOP_K = 5

# Visualization configuration
VIZ_THEME = 'plotly_white'
VIZ_HEIGHT = 600
VIZ_WIDTH = 1000

# ============================================================
# Lexical Resources
# ============================================================

# Domain-specific keywords for content analysis
DOMAIN_KEYWORDS = {
    'English': {
        'positive_emotion': ['happy', 'good', 'well', 'fine', 'excited', 'proud', 'blessed', 'grateful'],
        'negative_emotion': ['sad', 'worried', 'scared', 'anxious', 'stressed', 'pain', 'difficult', 'fear'],
        'healthcare': ['hospital', 'clinic', 'doctor', 'nurse', 'medicine', 'treatment', 'check-up', 'health worker'],
        'traditional': ['herb', 'traditional', 'local', 'cultural', 'belief', 'practice', 'remedy'],
        'support': ['husband', 'family', 'friend', 'mother', 'sister', 'relative', 'support'],
        'symptom': ['pain', 'swelling', 'nausea', 'bleeding', 'fever', 'cough', 'headache', 'fatigue']
    },
    'Swahili': {
        'positive_emotion': ['furaha', 'vizuri', 'nzuri', 'sawa', 'shangwe', 'fahari', 'baraka'],
        'negative_emotion': ['huzuni', 'wasiwasi', 'hofu', 'woga', 'msongo', 'maumivu', 'shida'],
        'healthcare': ['hospitali', 'kliniki', 'daktari', 'muuguzi', 'dawa', 'matibabu', 'mhudumu'],
        'traditional': ['mitishamba', 'kienyeji', 'mila', 'tamaduni', 'imani', 'dawa za jadi'],
        'support': ['mume', 'familia', 'rafiki', 'mama', 'dada', 'jamaa'],
        'symptom': ['maumivu', 'uvimbe', 'kichefuchefu', 'kutokwa damu', 'homa', 'kikohozi']
    },
    'Luganda': {
        'positive_emotion': ['essanyu', 'bulungi', 'nungi', 'musanyufu', 'nenyumiriza', 'mukisa'],
        'negative_emotion': ['kunisikiriza', 'kweraliikirira', 'okutya', 'obulumi', 'akabi', 'obuzibu'],
        'healthcare': ['ddwaaliro', 'kkanisa', 'musawo', 'ddagala', 'obujjanjabi', 'omusawo'],
        'traditional': ['ddagala lyekinnansi', 'omuddo', 'ennono', 'buwangwa', 'enzikiriza'],
        'support': ['baze', 'abaka', 'mukwano', 'maama', 'muganda', 'omwana'],
        'symptom': ['obulumi', 'okuzimba', 'okuziyira', 'okuyiwa omusaayi', 'omusujja']
    },
    'Runyankore': {
        'positive_emotion': ['okushemererwa', 'kurungi', 'marungi', 'nshemereirwe', 'omugisha'],
        'negative_emotion': ['okwerarikirira', 'okutiina', 'obusaasi', 'oburemeezi', 'akabi'],
        'healthcare': ['eirwariro', 'omushaho', 'omubazi', 'obujanjabi', 'okujanjaba'],
        'traditional': ['emibazi yobuhangwa', 'ebya buhangwa', 'emigyenzo', 'enyikiriza'],
        'support': ['omushaija', 'abantu', 'munywani', 'maama', 'mwenemaawe'],
        'symptom': ['obusaasi', 'okuzimba', 'okusheshemukwa', 'omushwija', 'okukorora']
    }
}

# ============================================================
# Visualization Configuration
# ============================================================

PLOTLY_TEMPLATES = ['plotly', 'plotly_white', 'plotly_dark']
COLOR_PALETTES = {
    'main': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
    'sequential': ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725'],
    'diverging': ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
}

# File naming
FIGURE_FORMAT = 'png'
FIGURE_DPI = 150