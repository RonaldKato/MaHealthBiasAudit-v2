"""
Bias Tracker Module
Tracks bias metrics before and after interventions/retraining
Enables comparison of bias across iterations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from config import THRESHOLDS, MODEL_CONFIGS, OUTPUT_DIR


@dataclass
class BiasSnapshot:
    """Snapshot of bias metrics at a point in time"""
    timestamp: str
    iteration: int
    intervention_description: str
    
    # Bias metrics
    avg_sdi: float
    avg_f1_disparity: float
    lang_cluster_purity: float
    topic_cluster_purity: float
    avg_fertility_penalty: float
    avg_oov_rate: float
    avg_morphological_alignment: float
    
    # Per-language metrics
    language_metrics: Dict[str, Dict] = field(default_factory=dict)
    
    # Per-topic metrics
    topic_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Flagged issues
    flags: List[str] = field(default_factory=list)


@dataclass
class InterventionResult:
    """Result of a bias reduction intervention"""
    intervention_name: str
    before_snapshot: BiasSnapshot
    after_snapshot: BiasSnapshot
    improvement_scores: Dict[str, float]
    success: bool
    recommendations: List[str]


class BiasTracker:
    """
    Tracks bias metrics across model iterations, interventions, and retraining
    Enables comparison of bias before and after improvements
    """
    
    def __init__(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(OUTPUT_DIR, "bias_tracker_history.json")
        
        self.save_path = save_path
        self.history: List[BiasSnapshot] = []
        self.interventions: List[InterventionResult] = []
        self.current_iteration = 0
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Load existing history if available
        if os.path.exists(save_path):
            self._load_history()
        
        print(f"✅ BiasTracker initialized (save_path: {save_path})")
    
    def take_snapshot(self, 
                      intervention_description: str,
                      sdi_matrix: pd.DataFrame,
                      performance_df: pd.DataFrame,
                      tokenisation_df: pd.DataFrame,
                      morph_results: Dict[str, pd.DataFrame],
                      embeddings: np.ndarray,
                      language_labels: List[str],
                      topic_labels: List[str]) -> BiasSnapshot:
        """
        Take a comprehensive bias snapshot at current state
        """
        self.current_iteration += 1
        
        # Compute average metrics
        # SDI - extract upper triangle values
        if sdi_matrix is not None and not sdi_matrix.empty:
            sdi_values = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)]
            avg_sdi = np.mean(sdi_values) if len(sdi_values) > 0 else 0.5
        else:
            avg_sdi = 0.5
        
        # F1 disparity
        if performance_df is not None and 'f1_disparity' in performance_df.columns:
            avg_f1_disparity = performance_df['f1_disparity'].mean()
        else:
            avg_f1_disparity = 0.0
        
        # Cluster purity
        lang_purity = self._compute_cluster_purity(embeddings, language_labels, n_clusters=4) if len(embeddings) > 0 else 0.5
        topic_purity = self._compute_cluster_purity(embeddings, topic_labels, n_clusters=5) if len(embeddings) > 0 else 0.5
        
        # Tokenisation metrics
        if tokenisation_df is not None and not tokenisation_df.empty:
            avg_fertility = tokenisation_df['Fertility_Penalty'].mean() if 'Fertility_Penalty' in tokenisation_df.columns else 1.5
            avg_oov = tokenisation_df['OOV_Rate'].mean() if 'OOV_Rate' in tokenisation_df.columns else 0.15
        else:
            avg_fertility = 1.5
            avg_oov = 0.15
        
        # Morphological alignment
        morph_scores = []
        if morph_results:
            for df in morph_results.values():
                if df is not None and 'boundary_f1' in df.columns:
                    morph_scores.extend(df['boundary_f1'].tolist())
        avg_morph = np.mean(morph_scores) if morph_scores else 0.6
        
        # Per-language metrics
        language_metrics = {}
        for lang in set(language_labels):
            lang_indices = [i for i, label in enumerate(language_labels) if label == lang]
            if lang_indices and len(embeddings) > max(lang_indices):
                lang_emb = embeddings[lang_indices]
                language_metrics[lang] = {
                    'embedding_mean_norm': float(np.mean(np.linalg.norm(lang_emb, axis=1))),
                    'n_samples': len(lang_indices)
                }
        
        # Per-topic metrics
        topic_metrics = {}
        for topic in set(topic_labels):
            topic_indices = [i for i, t in enumerate(topic_labels) if t == topic]
            topic_metrics[topic] = len(topic_indices)
        
        # Generate flags
        flags = []
        if avg_sdi > THRESHOLDS['sdi_high']:
            flags.append("HIGH_SDI")
        if avg_f1_disparity > THRESHOLDS['f1_disparity_high']:
            flags.append("HIGH_PERFORMANCE_DISPARITY")
        if avg_fertility > THRESHOLDS['tokenisation_parity']:
            flags.append("HIGH_FERTILITY_PENALTY")
        if avg_oov > THRESHOLDS['oov_rate']:
            flags.append("HIGH_OOV_RATE")
        if avg_morph < THRESHOLDS['mas_threshold']:
            flags.append("POOR_MORPHOLOGICAL_ALIGNMENT")
        if lang_purity > 0.6:
            flags.append("LANGUAGE_CLUSTERING_DETECTED")
        
        snapshot = BiasSnapshot(
            timestamp=datetime.now().isoformat(),
            iteration=self.current_iteration,
            intervention_description=intervention_description,
            avg_sdi=avg_sdi,
            avg_f1_disparity=avg_f1_disparity,
            lang_cluster_purity=lang_purity,
            topic_cluster_purity=topic_purity,
            avg_fertility_penalty=avg_fertility,
            avg_oov_rate=avg_oov,
            avg_morphological_alignment=avg_morph,
            language_metrics=language_metrics,
            topic_metrics=topic_metrics,
            flags=flags
        )
        
        self.history.append(snapshot)
        self._save_history()
        
        # Print snapshot summary
        print(f"\n📸 Bias Snapshot #{self.current_iteration}: {intervention_description}")
        print(f"   SDI: {avg_sdi:.3f} | F1 Disparity: {avg_f1_disparity:.3f} | "
              f"Lang Purity: {lang_purity:.3f}")
        print(f"   Fertility: {avg_fertility:.2f} | OOV: {avg_oov:.3f} | MAS: {avg_morph:.3f}")
        if flags:
            print(f"   Flags: {', '.join(flags)}")
        
        return snapshot
    
    def _compute_cluster_purity(self, 
                                 embeddings: np.ndarray,
                                 labels: List[str],
                                 n_clusters: int) -> float:
        """Compute cluster purity for given labels"""
        if len(embeddings) == 0 or len(embeddings) < n_clusters:
            return 0.5
        
        from sklearn.cluster import KMeans
        
        n_clusters = min(n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute purity
        purity = 0
        for cluster in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            
            cluster_true_labels = [labels[i] for i in cluster_indices if i < len(labels)]
            if not cluster_true_labels:
                continue
                
            label_counts = {}
            for label in cluster_true_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            max_label_count = max(label_counts.values()) if label_counts else 0
            purity += max_label_count
        
        return purity / max(len(embeddings), 1)
    
    def register_intervention(self, 
                               intervention_name: str,
                               before_snapshot: BiasSnapshot,
                               after_snapshot: BiasSnapshot,
                               target_metrics: List[str] = None) -> InterventionResult:
        """
        Register an intervention and compute improvements
        """
        if target_metrics is None:
            target_metrics = ['avg_sdi', 'avg_f1_disparity', 'avg_fertility_penalty', 
                             'avg_oov_rate', 'avg_morphological_alignment']
        
        improvement_scores = {}
        for metric in target_metrics:
            before_val = getattr(before_snapshot, metric, None)
            after_val = getattr(after_snapshot, metric, None)
            
            if before_val is not None and after_val is not None and before_val > 0:
                # For metrics where lower is better
                if metric in ['avg_sdi', 'avg_f1_disparity', 'avg_fertility_penalty', 'avg_oov_rate']:
                    improvement = (before_val - after_val) / before_val
                else:
                    improvement = (after_val - before_val) / before_val
                improvement_scores[metric] = improvement
        
        # Determine success (average improvement > 5%)
        valid_scores = [v for v in improvement_scores.values() if v is not None]
        avg_improvement = np.mean(valid_scores) if valid_scores else 0
        success = avg_improvement > 0.05
        
        # Generate recommendations
        recommendations = []
        if after_snapshot.avg_sdi > THRESHOLDS['sdi_high']:
            recommendations.append("SDI remains high - consider morphological tokenisation improvements")
        if after_snapshot.avg_fertility_penalty > THRESHOLDS['tokenisation_parity']:
            recommendations.append("Fertility penalty still elevated - implement MorphBPE")
        if after_snapshot.avg_oov_rate > THRESHOLDS['oov_rate']:
            recommendations.append("OOV rate remains high - augment vocabulary with domain terms")
        if after_snapshot.avg_morphological_alignment < THRESHOLDS['mas_threshold']:
            recommendations.append("Poor morphological alignment - consider morpheme-aware tokenisation")
        
        result = InterventionResult(
            intervention_name=intervention_name,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            improvement_scores=improvement_scores,
            success=success,
            recommendations=recommendations
        )
        
        self.interventions.append(result)
        self._save_history()
        
        print(f"\n🔄 Intervention: {intervention_name}")
        print(f"   Success: {'✅ YES' if success else '❌ NO'}")
        print(f"   Improvements: {', '.join([f'{k}: {v*100:+.1f}%' for k, v in improvement_scores.items()])}")
        
        return result
    
    def compare_snapshots(self, snapshot1: BiasSnapshot, snapshot2: BiasSnapshot) -> pd.DataFrame:
        """Compare two bias snapshots"""
        comparison = {
            'Metric': [],
            'Before': [],
            'After': [],
            'Change': [],
            'Improvement (%)': []
        }
        
        metrics = ['avg_sdi', 'avg_f1_disparity', 'lang_cluster_purity', 
                  'topic_cluster_purity', 'avg_fertility_penalty', 'avg_oov_rate',
                  'avg_morphological_alignment']
        
        for metric in metrics:
            before_val = getattr(snapshot1, metric, None)
            after_val = getattr(snapshot2, metric, None)
            
            if before_val is not None and after_val is not None:
                change = after_val - before_val
                # For metrics where lower is better
                if metric in ['avg_sdi', 'avg_f1_disparity', 'avg_fertility_penalty', 'avg_oov_rate']:
                    improvement_pct = (before_val - after_val) / max(abs(before_val), 0.001) * 100
                else:
                    improvement_pct = (after_val - before_val) / max(abs(before_val), 0.001) * 100
                
                comparison['Metric'].append(metric)
                comparison['Before'].append(f"{before_val:.4f}")
                comparison['After'].append(f"{after_val:.4f}")
                comparison['Change'].append(f"{change:+.4f}")
                comparison['Improvement (%)'].append(f"{improvement_pct:+.1f}%")
        
        return pd.DataFrame(comparison)
    
    def generate_progress_report(self) -> Dict:
        """Generate comprehensive progress report"""
        if len(self.history) < 2:
            return {"status": "Insufficient data", "message": "Need at least 2 snapshots"}
        
        first = self.history[0]
        last = self.history[-1]
        
        # Compute overall improvements
        improvements = {
            'SDI': (first.avg_sdi - last.avg_sdi) / max(abs(first.avg_sdi), 0.001) * 100,
            'F1_Disparity': (first.avg_f1_disparity - last.avg_f1_disparity) / max(abs(first.avg_f1_disparity), 0.001) * 100,
            'Fertility_Penalty': (first.avg_fertility_penalty - last.avg_fertility_penalty) / max(abs(first.avg_fertility_penalty), 0.001) * 100,
            'OOV_Rate': (first.avg_oov_rate - last.avg_oov_rate) / max(abs(first.avg_oov_rate), 0.001) * 100,
            'Morphological_Alignment': (last.avg_morphological_alignment - first.avg_morphological_alignment) / max(abs(first.avg_morphological_alignment), 0.001) * 100
        }
        
        # Determine overall trajectory
        if improvements['SDI'] > 10 and improvements['F1_Disparity'] > 10:
            trajectory = "POSITIVE - Bias significantly reduced"
        elif improvements['SDI'] > 0 or improvements['F1_Disparity'] > 0:
            trajectory = "MODERATE - Some improvement observed"
        else:
            trajectory = "CONCERNING - Bias metrics not improving"
        
        # Generate recommendations
        recommendations = []
        if last.avg_sdi > 0.3:
            recommendations.append("Consider implementing SDI-guided contrastive training")
        if last.avg_fertility_penalty > 1.5:
            recommendations.append("Implement MorphBPE for morphologically complex languages")
        if last.lang_cluster_purity > 0.6:
            recommendations.append("Add cross-lingual contrastive loss to reduce language clustering")
        
        return {
            'start_iteration': first.iteration,
            'end_iteration': last.iteration,
            'start_time': first.timestamp,
            'end_time': last.timestamp,
            'improvements': improvements,
            'trajectory': trajectory,
            'current_flags': last.flags,
            'recommendations': recommendations,
            'interventions_performed': len(self.interventions),
            'successful_interventions': sum(1 for i in self.interventions if i.success)
        }
    
    def _save_history(self):
        """Save history to JSON file"""
        history_data = []
        for snapshot in self.history:
            history_data.append({
                'timestamp': snapshot.timestamp,
                'iteration': snapshot.iteration,
                'intervention_description': snapshot.intervention_description,
                'avg_sdi': snapshot.avg_sdi,
                'avg_f1_disparity': snapshot.avg_f1_disparity,
                'lang_cluster_purity': snapshot.lang_cluster_purity,
                'topic_cluster_purity': snapshot.topic_cluster_purity,
                'avg_fertility_penalty': snapshot.avg_fertility_penalty,
                'avg_oov_rate': snapshot.avg_oov_rate,
                'avg_morphological_alignment': snapshot.avg_morphological_alignment,
                'flags': snapshot.flags
            })
        
        with open(self.save_path, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def _load_history(self):
        """Load history from JSON file"""
        with open(self.save_path, 'r') as f:
            history_data = json.load(f)
        
        self.history = []
        for data in history_data:
            snapshot = BiasSnapshot(
                timestamp=data['timestamp'],
                iteration=data['iteration'],
                intervention_description=data['intervention_description'],
                avg_sdi=data['avg_sdi'],
                avg_f1_disparity=data['avg_f1_disparity'],
                lang_cluster_purity=data['lang_cluster_purity'],
                topic_cluster_purity=data['topic_cluster_purity'],
                avg_fertility_penalty=data['avg_fertility_penalty'],
                avg_oov_rate=data['avg_oov_rate'],
                avg_morphological_alignment=data['avg_morphological_alignment'],
                flags=data['flags']
            )
            self.history.append(snapshot)
            
        if self.history:
            self.current_iteration = self.history[-1].iteration
    
    def get_summary(self) -> Dict:
        """Get a summary of the bias tracking history"""
        if not self.history:
            return {'status': 'No snapshots recorded'}
        
        return {
            'total_snapshots': len(self.history),
            'total_interventions': len(self.interventions),
            'first_snapshot': self.history[0].timestamp,
            'last_snapshot': self.history[-1].timestamp,
            'latest_metrics': {
                'avg_sdi': self.history[-1].avg_sdi,
                'avg_f1_disparity': self.history[-1].avg_f1_disparity,
                'avg_fertility_penalty': self.history[-1].avg_fertility_penalty,
                'avg_oov_rate': self.history[-1].avg_oov_rate,
                'lang_cluster_purity': self.history[-1].lang_cluster_purity
            },
            'latest_flags': self.history[-1].flags
        }


# Test the tracker
if __name__ == "__main__":
    tracker = BiasTracker()
    
    # Create dummy snapshot
    snapshot = BiasSnapshot(
        timestamp=datetime.now().isoformat(),
        iteration=1,
        intervention_description="Baseline",
        avg_sdi=0.45,
        avg_f1_disparity=0.25,
        lang_cluster_purity=0.68,
        topic_cluster_purity=0.72,
        avg_fertility_penalty=1.8,
        avg_oov_rate=0.18,
        avg_morphological_alignment=0.55,
        flags=["HIGH_SDI", "HIGH_FERTILITY_PENALTY"]
    )
    
    tracker.history.append(snapshot)
    print("\n✅ BiasTracker test complete!")