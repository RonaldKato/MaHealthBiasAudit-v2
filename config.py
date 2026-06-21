"""
MaHealthBiasAudit - Configuration Module
Centralized configuration for the entire bias audit system
"""

import os
from datetime import datetime

# ============================================================
# Basic Configuration
# ============================================================

PRIMARY_LANGUAGES = ['English', 'Swahili', 'Luganda', 'Runyankore']
MOTHER_LANGUAGES = ['English', 'Luganda', 'Runyankole-Rukiga']

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
REPORTS_DIR = os.path.join(OUTPUT_DIR, 'reports')
LOGS_DIR = os.path.join(OUTPUT_DIR, 'logs')
VALIDATION_DIR = os.path.join(OUTPUT_DIR, 'validation')
EXPERIMENTS_DIR = os.path.join(OUTPUT_DIR, 'experiments')
SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'samples')

for directory in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, LOGS_DIR, VALIDATION_DIR, EXPERIMENTS_DIR, SAMPLES_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================
# Encoder Configuration - LOCKED
# ============================================================

# Primary encoder: LaBSE (Language-agnostic BERT Sentence Embedding)
ENCODER_NAME = 'LaBSE'
ENCODER_DIM = 768
ENCODER_MODEL = 'sentence-transformers/LaBSE'

# Calibration thresholds
EQUIVALENCE_FLOOR = 0.15  # SDI for known-equivalent pairs
HIGH_BIAS_THRESHOLD = 0.40  # Calibrated high-bias threshold
DIVERGENCE_CEILING = 0.90  # SDI for non-matching pairs

# Reference sets for calibration
CALIBRATION_REFERENCE_SETS = {
    'paraphrase_pairs': 'English paraphrase-paraphrase pairs (equivalence floor)',
    'verified_pairs': 'Native-speaker-validated English-Luganda/Runyankore/Swahili pairs',
    'non_matching_pairs': 'Deliberately non-matching pairs (divergence ceiling)'
}

# ============================================================
# Feature Attribution Configuration
# ============================================================

FEATURE_NAMES = [
    'subword_fertility',
    'agglutinative_verb_complex_depth',
    'clinical_loanword_count',
    'negation',
    'dosage_numeric_expressions',
    'concord_chain_length',
    'sentence_length'
]

FEATURE_ORDER = {
    'subword_fertility': 0,
    'agglutinative_verb_complex_depth': 1,
    'clinical_loanword_count': 2,
    'negation': 3,
    'dosage_numeric_expressions': 4,
    'concord_chain_length': 5,
    'sentence_length': 6
}

# ============================================================
# Bias Reduction Configuration
# ============================================================

BIAS_REDUCTION_INTERVENTIONS = {
    'content_omission': {
        'name': 'Clinical-term anchoring + completeness',
        'description': 'Add missing clinical terms and complete the response'
    },
    'tokenization_subword': {
        'name': 'Morphology-aware normalization',
        'description': 'Normalize subword tokens to improve readability'
    },
    'negation_reversal': {
        'name': 'Consistency re-ranking preserving negation',
        'description': 'Ensure negation is preserved in translation'
    },
    'length_disparity': {
        'name': 'Length parity restoration',
        'description': 'Add missing content to match English length'
    },
    'cultural_mismatch': {
        'name': 'Cultural perspective balancing',
        'description': 'Include both traditional and medical perspectives'
    }
}

# ============================================================
# Experiment Configuration
# ============================================================

EXPERIMENT_SIZES = [10, 100, 1000, 10000]

# Performance tracking
TRACK_METRICS = [
    'avg_sdi',
    'bias_level',
    'total_flags',
    'execution_time',
    'avg_response_length',
    'vocabulary_richness',
    'fertility_penalty'
]

# ============================================================
# Execution Configuration
# ============================================================

RANDOM_SEED = 42
EXECUTION_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================
# Analysis Configuration
# ============================================================

SIGNIFICANCE_ALPHA = 0.05
MIN_ANSWER_LENGTH = 5
MAX_ANSWER_LENGTH = 5000

MAX_TOKEN_FERTILITY = 1.5
SENTENCE_PIECING_THRESHOLD = 0.3
WORD_PIECING_THRESHOLD = 0.2

EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128

SDI_THRESHOLD_HIGH = 0.4
SDI_THRESHOLD_MODERATE = 0.2
RCA_TOP_K = 5

# ============================================================
# Bias Sentence Characteristics - ENHANCED with structures
# ============================================================

BIAS_SENTENCE_CHARACTERISTICS = {
    'translation_length_disparity': {
        'description': 'Sentences where non-English responses are significantly shorter, losing content',
        'structure': 'English contains multiple clauses and details; translation contains only main clause',
        'example': 'English: "You should visit the hospital immediately if you experience severe pain or unusual bleeding during pregnancy."\nSwahili: "Tembelea hospitali kwa maumivu makali." (Severely truncated)',
        'detection': 'Length ratio < 0.6 compared to English',
        'severity': 'High',
        'mitigation': 'Ensure complete translation of all medical information'
    },
    'cultural_concept_mismatch': {
        'description': 'Where traditional practices replace or conflict with medical advice',
        'structure': 'Medical advice in English replaced with traditional remedy in target language',
        'example': 'English: "Consult your doctor about pain management options."\nRunyankore: "Koresha emibazi y\'obuhangwa ku mishija." (Use traditional herbs for pain)',
        'detection': 'Presence of traditional keywords without medical equivalents',
        'severity': 'High',
        'mitigation': 'Include both traditional and medical perspectives'
    },
    'medical_term_omission': {
        'description': 'Critical medical terms missing in low-resource languages',
        'structure': 'Technical medical terms in English are either omitted or replaced with generic terms',
        'example': 'English: "Monitor for preeclampsia symptoms including high blood pressure and swelling."\nLuganda: "Laba obulumi n\'okuzimba." (Missing "preeclampsia" and "high blood pressure")',
        'detection': 'Domain keyword absence in non-English responses',
        'severity': 'Critical',
        'mitigation': 'Create standardized medical terminology for each language'
    },
    'emotional_tone_shift': {
        'description': 'Urgent/empathetic tone in English becomes neutral or dismissive',
        'structure': 'Emphatic adjectives and urgency markers replaced with neutral phrasing',
        'example': 'English: "This is very serious - please seek immediate medical attention!"\nSwahili: "Unaweza kwenda hospitali." (You can go to hospital)',
        'detection': 'Reduction in emotion-related keywords',
        'severity': 'Moderate',
        'mitigation': 'Train translators on preserving emotional tone'
    },
    'structural_simplification': {
        'description': 'Complex sentence structures simplified, losing nuance',
        'structure': 'Conditional clauses ("although", "if", "when") and qualifiers removed',
        'example': 'English: "Although herbal remedies may be traditional, you should always consult your healthcare provider first."\nLuganda: "Kozesa ddagala lyekinnansi." (Use traditional medicine)',
        'detection': 'Loss of conditional clauses and qualifiers',
        'severity': 'Moderate',
        'mitigation': 'Preserve conditional and advisory structures'
    },
    'pronoun_ambiguity': {
        'description': 'Ambiguous pronoun references leading to confusion about subject',
        'structure': 'Pronouns without clear antecedents in target language',
        'example': 'English: "The mother should take her medication as prescribed by her doctor."\nSwahili: "Anapaswa kuchukua dawa yake." (Unclear who "anapaswa" refers to)',
        'detection': 'Multiple possible antecedents for pronouns',
        'severity': 'Moderate',
        'mitigation': 'Use explicit noun phrases instead of pronouns'
    },
    'instructional_generic_shift': {
        'description': 'Specific instructions become generic or vague',
        'structure': 'Concrete numbers, frequencies, and timings omitted',
        'example': 'English: "Take 500mg of paracetamol every 6 hours for 3 days."\nLuganda: "Funa eddagala ly\'omusaayi." (Get medicine for blood)',
        'detection': 'Absence of numerical and temporal specifics',
        'severity': 'High',
        'mitigation': 'Include specific dosages and timings in all languages'
    },
    'causality_inversion': {
        'description': 'Cause-and-effect relationships reversed or lost',
        'structure': 'Causal connectors ("because", "therefore", "since") missing or misused',
        'example': 'English: "Because you have high blood pressure, you need to avoid salty foods."\nRunyankore: "Ota omunyu." (Avoid salt - missing causal link)',
        'detection': 'Loss of causal connectives and explanations',
        'severity': 'High',
        'mitigation': 'Preserve causal relationships in translation'
    },
    'negation_misinterpretation': {
        'description': 'Negations incorrectly translated or omitted',
        'structure': 'Negative constructions ("don\'t", "never", "avoid") become positive',
        'example': 'English: "Do not take this medication with alcohol."\nSwahili: "Unaweza kuchukua dawa hii na pombe." (You can take this medication with alcohol)',
        'detection': 'Loss or inversion of negation markers',
        'severity': 'Critical',
        'mitigation': 'Explicitly mark negations in all translations'
    }
}

# ============================================================
# Bias Reduction Templates
# ============================================================

BIAS_REDUCTION_TEMPLATES = {
    'translation_length_disparity': {
        'problem': 'Non-English responses are significantly shorter, losing content',
        'solution': 'Ensure complete translation with all clauses preserved',
        'template': 'Original: {original}\nIssue: Response is significantly shorter\nSolution: Add missing clauses and details\nDebiased: {debiased}'
    },
    'cultural_concept_mismatch': {
        'problem': 'Traditional practices replace or conflict with medical advice',
        'solution': 'Include both traditional and medical perspectives',
        'template': 'Original: {original}\nIssue: Traditional remedy replaces medical advice\nSolution: Present both traditional and clinical perspectives\nDebiased: {debiased}'
    },
    'medical_term_omission': {
        'problem': 'Critical medical terms missing in translation',
        'solution': 'Use standardized medical terminology',
        'template': 'Original: {original}\nIssue: Medical terms are missing\nSolution: Include correct medical terminology\nDebiased: {debiased}'
    },
    'emotional_tone_shift': {
        'problem': 'Urgent/empathetic tone becomes neutral or dismissive',
        'solution': 'Preserve emotional tone and urgency markers',
        'template': 'Original: {original}\nIssue: Loss of urgency/empathy\nSolution: Add emotional cues and urgency markers\nDebiased: {debiased}'
    },
    'structural_simplification': {
        'problem': 'Complex sentence structures simplified, losing nuance',
        'solution': 'Preserve conditional and advisory structures',
        'template': 'Original: {original}\nIssue: Conditional clauses and qualifiers removed\nSolution: Restore conditional and advisory structures\nDebiased: {debiased}'
    },
    'pronoun_ambiguity': {
        'problem': 'Ambiguous pronoun references',
        'solution': 'Use explicit noun phrases instead of pronouns',
        'template': 'Original: {original}\nIssue: Ambiguous pronoun reference\nSolution: Replace pronouns with clear noun phrases\nDebiased: {debiased}'
    },
    'instructional_generic_shift': {
        'problem': 'Specific instructions become generic or vague',
        'solution': 'Include all numerical and temporal specifics',
        'template': 'Original: {original}\nIssue: Missing specific dosages, frequencies, or timings\nSolution: Add all specific instructions\nDebiased: {debiased}'
    },
    'causality_inversion': {
        'problem': 'Cause-and-effect relationships reversed or lost',
        'solution': 'Preserve causal connectors and explanations',
        'template': 'Original: {original}\nIssue: Causal link missing or inverted\nSolution: Restore causal relationship\nDebiased: {debiased}'
    },
    'negation_misinterpretation': {
        'problem': 'Negations incorrectly translated or omitted',
        'solution': 'Explicitly mark negations',
        'template': 'Original: {original}\nIssue: Negation missing or inverted\nSolution: Use explicit negation markers\nDebiased: {debiased}'
    },
    'unknown': {
        'problem': 'Potential bias detected in translation',
        'solution': 'Review and improve translation quality',
        'template': 'Original: {original}\nIssue: Potential bias detected\nSolution: Review and improve translation\nDebiased: {debiased}'
    }
}

# ============================================================
# Lexical Resources - ENHANCED
# ============================================================

DOMAIN_KEYWORDS = {
    'English': {
        'positive_emotion': ['happy', 'good', 'well', 'fine', 'excited', 'proud', 'blessed', 'grateful', 'wonderful', 'great'],
        'negative_emotion': ['sad', 'worried', 'scared', 'anxious', 'stressed', 'pain', 'difficult', 'fear', 'afraid', 'concerned'],
        'healthcare': ['hospital', 'clinic', 'doctor', 'nurse', 'medicine', 'treatment', 'check-up', 'health worker', 'appointment', 'prescription'],
        'traditional': ['herb', 'traditional', 'local', 'cultural', 'belief', 'practice', 'remedy', 'herbal', 'healer'],
        'support': ['husband', 'family', 'friend', 'mother', 'sister', 'relative', 'support', 'community', 'neighbor'],
        'symptom': ['pain', 'swelling', 'nausea', 'bleeding', 'fever', 'cough', 'headache', 'fatigue', 'dizziness', 'vomiting'],
        'medical_terms': ['preeclampsia', 'hypertension', 'diabetes', 'anemia', 'infection', 'medication', 'dosage', 'ultrasound', 'vaccination']
    },
    'Swahili': {
        'positive_emotion': ['furaha', 'vizuri', 'nzuri', 'sawa', 'shangwe', 'fahari', 'baraka', 'heri', 'njema', 'radhi'],
        'negative_emotion': ['huzuni', 'wasiwasi', 'hofu', 'woga', 'msongo', 'maumivu', 'shida', 'taabu', 'hangaiko', 'dhiki'],
        'healthcare': ['hospitali', 'kliniki', 'daktari', 'muuguzi', 'dawa', 'matibabu', 'mhudumu', 'afya', 'zahanati', 'huduma'],
        'traditional': ['mitishamba', 'kienyeji', 'mila', 'tamaduni', 'imani', 'dawa za jadi', 'mganga', 'waganga', 'shaman'],
        'support': ['mume', 'familia', 'rafiki', 'mama', 'dada', 'jamaa', 'mzazi', 'ndugu', 'mwenzi', 'kijiji'],
        'symptom': ['maumivu', 'uvimbe', 'kichefuchefu', 'kutokwa damu', 'homa', 'kikohozi', 'kichwa', 'kizunguzungu', 'kutapika'],
        'medical_terms': ['bremba', 'shinikizo la damu', 'ugonjwa wa sukari', 'upungufu wa damu', 'maambukizi', 'dawa', 'kiwango cha dawa']
    },
    'Luganda': {
        'positive_emotion': ['essanyu', 'bulungi', 'nungi', 'musanyufu', 'nenyumiriza', 'mukisa', 'webale', 'mirembe', 'mpozzi'],
        'negative_emotion': ['kunisikiriza', 'kweraliikirira', 'okutya', 'obulumi', 'akabi', 'obuzibu', 'ennaku', 'okuteraliikirira'],
        'healthcare': ['ddwaaliro', 'kkanisa', 'musawo', 'ddagala', 'obujjanjabi', 'omusawo', 'ebyobulamu', 'kliniki', 'okujjanjaba'],
        'traditional': ['ddagala lyekinnansi', 'omuddo', 'ennono', 'buwangwa', 'enzikiriza', 'bajajja', 'okusamira', 'emiti'],
        'support': ['baze', 'abaka', 'mukwano', 'maama', 'muganda', 'omwana', 'kitange', 'jajja', 'ntalo'],
        'symptom': ['obulumi', 'okuzimba', 'okuziyira', 'okuyiwa omusaayi', 'omusujja', 'enkologi', 'okudduka omusulo'],
        'medical_terms': ['preeclampsia', 'omusaayi omungi', 'sukaali', 'okwonoonebwa', 'obulwadde', 'eddagala', 'omugereko']
    },
    'Runyankore': {
        'positive_emotion': ['okushemererwa', 'kurungi', 'marungi', 'nshemereirwe', 'omugisha', 'webare', 'omurungi', 'kuburungi'],
        'negative_emotion': ['okwerarikirira', 'okutiina', 'obusaasi', 'oburemeezi', 'akabi', 'obunaku', 'okweteganya', 'obunyi'],
        'healthcare': ['eirwariro', 'omushaho', 'omubazi', 'obujanjabi', 'okujanjaba', 'obujanjabi bw\'amaani', 'ecliniki', 'okureeba'],
        'traditional': ['emibazi yobuhangwa', 'ebya buhangwa', 'emigyenzo', 'enyikiriza', 'obuganga', 'omuganga', 'ebibazi'],
        'support': ['omushaija', 'abantu', 'munywani', 'maama', 'mwenemaawe', 'omwana', 'ishe', 'nyina', 'omuka'],
        'symptom': ['obusaasi', 'okuzimba', 'okusheshemukwa', 'omushwija', 'okukorora', 'omutwe', 'okujuba', 'okutaaha'],
        'medical_terms': ['preeclampsia', 'omusaayi gw\'amaani', 'omusujja gw\'omubiri', 'obulwadde', 'emibazi', 'omugereko']
    }
}

# ============================================================
# Visualization Configuration
# ============================================================

COLOR_PALETTES = {
    'main': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
    'sequential': ['#440154', '#3b528b', '#21908c', '#5dc963', '#fde725'],
    'diverging': ['#d73027', '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4']
}

FIGURE_DPI = 150
VIZ_HEIGHT = 600
VIZ_WIDTH = 1000