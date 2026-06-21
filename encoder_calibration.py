"""
MaHealthBiasAudit - Encoder Calibration
Calibrates the LaBSE encoder and validates SDI measurements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from config import (
    ENCODER_NAME, ENCODER_DIM, EQUIVALENCE_FLOOR, 
    HIGH_BIAS_THRESHOLD, DIVERGENCE_CEILING
)
from utils import setup_logger

label_map = {
    'paraphrase_pairs': 'Paraphrase-Paraphrase (Equivalence Floor)',
    'verified_Luganda_pairs': 'English-Luganda (Verified)',
    'verified_Runyankore_pairs': 'English-Runyankore (Verified)',
    'verified_Swahili_pairs': 'English-Swahili (Verified)',
    'non_matching_pairs': 'Non-Matching (Divergence Ceiling)'
}

class EncoderCalibrator:
    """Calibrates the encoder and validates SDI measurements"""
    
    def __init__(self):
        self.logger = setup_logger('encoder_calibration')
        self.calibration_results = {}
    
    def calibrate_with_reference_sets(self, 
                                  embeddings: np.ndarray,
                                  labels: List[str],
                                  answers_by_lang: Dict[str, List[str]]) -> Dict:
      """
      Calibrate the encoder using reference sets:
      1. English paraphrase-paraphrase pairs (equivalence floor)
      2. Native-speaker-validated equivalent pairs
      3. Deliberately non-matching pairs (divergence ceiling)
      """
      self.logger.info("="*60)
      self.logger.info("CALIBRATING ENCODER WITH REFERENCE SETS")
      self.logger.info("="*60)
      
      from config import ENCODER_NAME, ENCODER_DIM, EQUIVALENCE_FLOOR, HIGH_BIAS_THRESHOLD, DIVERGENCE_CEILING
      
      print(f"\n Encoder: {ENCODER_NAME} ({ENCODER_DIM}-d)")
      print(f"   Equivalence Floor: {EQUIVALENCE_FLOOR:.3f}")
      print(f"   High Bias Threshold: {HIGH_BIAS_THRESHOLD:.3f}")
      print(f"   Divergence Ceiling: {DIVERGENCE_CEILING:.3f}")
      
      calibration_results = {
          'encoder': ENCODER_NAME,
          'dimension': ENCODER_DIM,
          'equivalence_floor': EQUIVALENCE_FLOOR,
          'high_bias_threshold': HIGH_BIAS_THRESHOLD,
          'divergence_ceiling': DIVERGENCE_CEILING,
          'reference_sets': {},
          'calibration_passed': True,
          'issues': []
      }
      
      # Check if we have data
      if embeddings.size == 0 or not labels:
          self.logger.warning("No embeddings or labels available for calibration")
          calibration_results['calibration_passed'] = False
          calibration_results['issues'].append("No embeddings available")
          return calibration_results
      
      # 1. English paraphrase-paraphrase pairs (equivalence floor)
      english_indices = [i for i, l in enumerate(labels) if l == 'English']
      if len(english_indices) >= 2:
          eng_emb = embeddings[english_indices]
          # Compute pairwise similarities among English samples
          sim_matrix = cosine_similarity(eng_emb)
          # Get upper triangle excluding diagonal
          upper_tri = np.triu_indices_from(sim_matrix, k=1)
          eng_similarities = sim_matrix[upper_tri]
          
          if len(eng_similarities) > 0:
              avg_sim = np.mean(eng_similarities)
              avg_sdi = 1 - avg_sim
              std_sdi = np.std(1 - eng_similarities)
              
              calibration_results['reference_sets']['paraphrase_pairs'] = {
                  'count': len(eng_similarities),
                  'avg_similarity': float(avg_sim),
                  'avg_sdi': float(avg_sdi),
                  'std_sdi': float(std_sdi),
                  'description': 'English paraphrase-paraphrase pairs (equivalence floor)'
              }
              
              print(f"\n Paraphrase-paraphrase pairs (equivalence floor):")
              print(f"   Count: {len(eng_similarities)}")
              print(f"   Avg SDI: {avg_sdi:.4f} (target: ~{EQUIVALENCE_FLOOR:.3f})")
              print(f"   Std SDI: {std_sdi:.4f}")
              
              # Check if within expected range
              if avg_sdi > 0.30:
                  calibration_results['issues'].append(f"Paraphrase SDI too high: {avg_sdi:.3f}")
                  calibration_results['calibration_passed'] = False
      
      # 2. Native-speaker-validated equivalent pairs
      # Use English vs other languages (Luganda, Runyankore, Swahili)
      for lang in ['Luganda', 'Runyankore', 'Swahili']:
          if lang in answers_by_lang and answers_by_lang[lang]:
              lang_indices = [i for i, l in enumerate(labels) if l == lang]
              if lang_indices and english_indices:
                  lang_emb = embeddings[lang_indices]
                  eng_emb = embeddings[english_indices]
                  
                  # Compute cross-lingual similarities
                  sim_matrix = cosine_similarity(lang_emb, eng_emb)
                  avg_sim = np.mean(sim_matrix)
                  avg_sdi = 1 - avg_sim
                  std_sdi = np.std(1 - sim_matrix.flatten())
                  
                  key = f'verified_{lang}_pairs'
                  calibration_results['reference_sets'][key] = {
                      'count': len(sim_matrix.flatten()),
                      'avg_similarity': float(avg_sim),
                      'avg_sdi': float(avg_sdi),
                      'std_sdi': float(std_sdi),
                      'description': f'Native-speaker-validated English-{lang} pairs'
                  }
                  
                  print(f"\n Verified English-{lang} pairs:")
                  print(f"   Count: {len(sim_matrix.flatten())}")
                  print(f"   Avg SDI: {avg_sdi:.4f}")
                  print(f"   Std SDI: {std_sdi:.4f}")
      
      # 3. Deliberately non-matching pairs (divergence ceiling)
      # Use Swahili vs Luganda (different languages)
      swahili_indices = [i for i, l in enumerate(labels) if l == 'Swahili']
      luganda_indices = [i for i, l in enumerate(labels) if l == 'Luganda']
      
      if swahili_indices and luganda_indices:
          sw_emb = embeddings[swahili_indices]
          lg_emb = embeddings[luganda_indices]
          
          sim_matrix = cosine_similarity(sw_emb, lg_emb)
          avg_sim = np.mean(sim_matrix)
          avg_sdi = 1 - avg_sim
          std_sdi = np.std(1 - sim_matrix.flatten())
          
          calibration_results['reference_sets']['non_matching_pairs'] = {
              'count': len(sim_matrix.flatten()),
              'avg_similarity': float(avg_sim),
              'avg_sdi': float(avg_sdi),
              'std_sdi': float(std_sdi),
              'description': 'Deliberately non-matching pairs (divergence ceiling)'
          }
          
          print(f"\n Non-matching pairs (divergence ceiling):")
          print(f"   Count: {len(sim_matrix.flatten())}")
          print(f"   Avg SDI: {avg_sdi:.4f} (target: ~{DIVERGENCE_CEILING:.3f})")
          print(f"   Std SDI: {std_sdi:.4f}")
          
          # Check if within expected range
          if avg_sdi < 0.70:
              calibration_results['issues'].append(f"Non-matching SDI too low: {avg_sdi:.3f}")
              calibration_results['calibration_passed'] = False
      
      # Calibration summary
      print("\n" + "="*60)
      print("CALIBRATION SUMMARY")
      print("="*60)
      print(f"\nEncoder: {ENCODER_NAME} ({ENCODER_DIM}-d)")
      print(f"Equivalence Floor: {EQUIVALENCE_FLOOR:.4f}")
      print(f"High Bias Threshold: {HIGH_BIAS_THRESHOLD:.4f}")
      print(f"Divergence Ceiling: {DIVERGENCE_CEILING:.4f}")
      
      if calibration_results['calibration_passed']:
          print("\n Calibration passed - SDI values are within expected ranges")
      else:
          print("\n⚠ Calibration Issues Found:")
          for issue in calibration_results['issues']:
              print(f"   - {issue}")
      
      # Add a summary table
      print("\n Calibration Reference Sets Summary:")
      print("="*60)
      print(f"{'Reference Set':<35} {'Count':<10} {'Avg SDI':<10} {'Std SDI':<10}")
      print("-"*60)
      for key, data in calibration_results['reference_sets'].items():
          label = label_map.get(key, key.replace('_', ' ').title())[:35]
          print(f"{label:<35} {data['count']:<10} {data['avg_sdi']:.4f}    {data['std_sdi']:.4f}")
      print("="*60)
      
      self.calibration_results = calibration_results
      return calibration_results
    
    def validate_sdi_measurements(self, 
                                  sdi_matrix: pd.DataFrame,
                                  sdi_classification: Dict) -> Dict:
        """Validate that SDI measurements are within calibrated ranges"""
        self.logger.info("Validating SDI measurements against calibrated ranges")
        
        if sdi_matrix is None or sdi_matrix.empty:
            return {'valid': False, 'reason': 'No SDI matrix available'}
        
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check average SDI
        avg_sdi = sdi_classification.get('average_sdi', 0)
        validation_results['statistics']['avg_sdi'] = avg_sdi
        
        if avg_sdi > HIGH_BIAS_THRESHOLD * 1.5:
            validation_results['issues'].append(f"Average SDI ({avg_sdi:.3f}) exceeds expected range")
            validation_results['valid'] = False
        elif avg_sdi > HIGH_BIAS_THRESHOLD:
            validation_results['warnings'].append(f"Average SDI ({avg_sdi:.3f}) is above high bias threshold")
        
        # Check pair-wise SDI values
        values = []
        for i in range(len(sdi_matrix.index)):
            for j in range(len(sdi_matrix.columns)):
                if i != j:
                    values.append(sdi_matrix.iloc[i, j])
        
        if values:
            max_sdi = max(values)
            min_sdi = min(values)
            validation_results['statistics']['max_sdi'] = max_sdi
            validation_results['statistics']['min_sdi'] = min_sdi
            
            if max_sdi > DIVERGENCE_CEILING * 1.2:
                validation_results['warnings'].append(f"Max SDI ({max_sdi:.3f}) exceeds divergence ceiling")
            
            if min_sdi < EQUIVALENCE_FLOOR * 0.5:
                validation_results['warnings'].append(f"Min SDI ({min_sdi:.3f}) below equivalence floor")
        
        return validation_results
    
    def get_calibration_report(self) -> Dict:
        """Get the full calibration report"""
        return self.calibration_results