"""
Bias Tracker Module
Tracks bias metrics before and after interventions/retraining
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from config import THRESHOLDS, MODEL_CONFIGS

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
    language_metrics: Dict[str, Dict]
    
    # Per-topic metrics
    topic_metrics: Dict[str, float]
    
    # Flagged issues
    flags: List[str]

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
    
    def __init__(self, save_path: str = "bias_tracker_history.json"):
        self.save_path = save_path
        self.history: List[BiasSnapshot] = []
        self.interventions: List[InterventionResult] = []
        self.current_iteration = 0
        
        # Load existing history if available
        if os.path.exists(save_path):
            self._load_history()
    
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
        sdi_values = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)]
        avg_sdi = np.mean(sdi_values)
        
        # F1 disparity
        if 'f1_disparity' in performance_df.columns:
            avg_f1_disparity = performance_df['f1_disparity'].mean()
        else:
            avg_f1_disparity = 0.0
        
        # Cluster purity (requires existing computation)
        lang_purity = self._compute_cluster_purity(embeddings, language_labels, n_clusters=4)
        topic_purity = self._compute_cluster_purity(embeddings, topic_labels, n_clusters=5)
        
        # Tokenisation metrics
        avg_fertility = tokenisation_df['Fertility_Penalty'].mean() if not tokenisation_df.empty else 1.0
        avg_oov = tokenisation_df['OOV_Rate'].mean() if not tokenisation_df.empty else 0.0
        
        # Morphological alignment
        morph_scores = []
        for df in morph_results.values():
            if 'boundary_f1' in df.columns:
                morph_scores.extend(df['boundary_f1'].tolist())
        avg_morph = np.mean(morph_scores) if morph_scores else 0.0
        
        # Per-language metrics
        language_metrics = {}
        for lang in set(language_labels):
            # Get language-specific indices for embeddings
            lang_indices = [i for i, label in enumerate(language_labels) if label == lang]
            if lang_indices:
                lang_emb = embeddings[lang_indices]
                lang_metrics = {
                    'embedding_mean_norm': np.mean(np.linalg.norm(lang_emb, axis=1)),
                    'n_samples': len(lang_indices)
                }
                language_metrics[lang] = lang_metrics
        
        # Per-topic metrics
        topic_metrics = {}
        for topic in set(topic_labels):
            topic_indices = [i for i, t in enumerate(topic_labels) if t == topic]
            topic_metrics[topic] = len(topic_indices)
        
        # Generate flags
        flags = []
        if avg_sdi > THRESHOLDS['sdi_high']:
            flags.append("HIGH_SDI")
        if avg_f1_disparity > 0.3:
            flags.append("HIGH_PERFORMANCE_DISPARITY")
        if avg_fertility > THRESHOLDS['tokenisation_parity']:
            flags.append("HIGH_FERTILITY_PENALTY")
        if avg_oov > THRESHOLDS['oov_rate']:
            flags.append("HIGH_OOV_RATE")
        if avg_morph < THRESHOLDS['mas_threshold']:
            flags.append("POOR_MORPHOLOGICAL_ALIGNMENT")
        if lang_purity > 0.7:
            flags.append("STRONG_LANGUAGE_CLUSTERING")
        
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
        
        return snapshot
    
    def _compute_cluster_purity(self, 
                                 embeddings: np.ndarray,
                                 labels: List[str],
                                 n_clusters: int) -> float:
        """Compute cluster purity for given labels"""
        from sklearn.cluster import KMeans
        
        if len(embeddings) == 0:
            return 0.0
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), 
                        random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute purity
        purity = 0
        for cluster in np.unique(cluster_labels):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            if len(cluster_indices) == 0:
                continue
            
            cluster_true_labels = [labels[i] for i in cluster_indices]
            label_counts = {}
            for label in cluster_true_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            max_label_count = max(label_counts.values()) if label_counts else 0
            purity += max_label_count
        
        return purity / len(embeddings)
    
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
            
            if before_val is not None and after_val is not None:
                # For metrics where lower is better
                if metric in ['avg_sdi', 'avg_f1_disparity', 'avg_fertility_penalty', 'avg_oov_rate']:
                    improvement = (before_val - after_val) / max(before_val, 0.001)
                else:
                    improvement = (after_val - before_val) / max(before_val, 0.001)
                improvement_scores[metric] = improvement
        
        # Determine success (average improvement > 5%)
        avg_improvement = np.mean(list(improvement_scores.values())) if improvement_scores else 0
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
                    improvement_pct = (before_val - after_val) / max(before_val, 0.001) * 100
                else:
                    improvement_pct = (after_val - before_val) / max(before_val, 0.001) * 100
                
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
            'SDI': (first.avg_sdi - last.avg_sdi) / max(first.avg_sdi, 0.001) * 100,
            'F1_Disparity': (first.avg_f1_disparity - last.avg_f1_disparity) / max(first.avg_f1_disparity, 0.001) * 100,
            'Fertility_Penalty': (first.avg_fertility_penalty - last.avg_fertility_penalty) / max(first.avg_fertility_penalty, 0.001) * 100,
            'OOV_Rate': (first.avg_oov_rate - last.avg_oov_rate) / max(first.avg_oov_rate, 0.001) * 100,
            'Morphological_Alignment': (last.avg_morphological_alignment - first.avg_morphological_alignment) / max(first.avg_morphological_alignment, 0.001) * 100
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
    
    def visualize_progress(self, save_path: Optional[str] = None) -> None:
        """Create visualization of bias progress over time"""
        import matplotlib.pyplot as plt
        
        if len(self.history) < 2:
            print("Need at least 2 snapshots for visualization")
            return
        
        # Extract data
        iterations = [s.iteration for s in self.history]
        metrics = {
            'SDI': [s.avg_sdi for s in self.history],
            'F1 Disparity': [s.avg_f1_disparity for s in self.history],
            'Fertility Penalty': [s.avg_fertility_penalty for s in self.history],
            'OOV Rate': [s.avg_oov_rate for s in self.history],
            'Morphological Alignment': [s.avg_morphological_alignment for s in self.history],
            'Language Purity': [s.lang_cluster_purity for s in self.history]
        }
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]
            ax.plot(iterations, values, marker='o', linewidth=2, markersize=8)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            
            # Add threshold lines where applicable
            if metric_name == 'SDI':
                ax.axhline(y=0.4, color='red', linestyle='--', label='High threshold')
                ax.axhline(y=0.2, color='orange', linestyle='--', label='Moderate threshold')
                ax.legend()
            elif metric_name == 'F1 Disparity':
                ax.axhline(y=0.3, color='red', linestyle='--', label='High disparity')
                ax.legend()
            elif metric_name == 'Fertility Penalty':
                ax.axhline(y=1.5, color='red', linestyle='--', label='High fertility')
                ax.legend()
            elif metric_name == 'OOV Rate':
                ax.axhline(y=0.15, color='red', linestyle='--', label='High OOV')
                ax.legend()
            elif metric_name == 'Morphological Alignment':
                ax.axhline(y=0.6, color='green', linestyle='--', label='Target')
                ax.legend()
        
        # Hide unused subplot
        axes[-1].set_visible(False)
        
        plt.suptitle('Bias Metrics Progress Over Iterations', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def _save_history(self):
        """Save history to JSON file"""
        # Convert snapshots to serializable format
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
                language_metrics={},
                topic_metrics={},
                flags=data['flags']
            )
            self.history.append(snapshot)
            
        if self.history:
            self.current_iteration = self.history[-1].iteration


# Example usage
if __name__ == "__main__":
    # Create bias tracker
    tracker = BiasTracker(save_path="bias_tracker_demo.json")
    
    # Create dummy data for snapshots
    np.random.seed(42)
    
    # Create dummy SDI matrix
    languages = ['English', 'Swahili', 'Yoruba', 'Amharic']
    sdi_matrix = pd.DataFrame(
        np.array([[0, 0.35, 0.45, 0.55],
                  [0.35, 0, 0.30, 0.48],
                  [0.45, 0.30, 0, 0.32],
                  [0.55, 0.48, 0.32, 0]]),
        index=languages, columns=languages
    )
    
    # Dummy performance dataframe
    performance_df = pd.DataFrame({
        'model_name': ['mBERT', 'mBERT', 'mBERT'],
        'language': ['English', 'Swahili', 'Yoruba'],
        'f1_disparity': [0, 0.25, 0.35],
        'exact_match': [0.85, 0.65, 0.55]
    })
    
    # Dummy tokenisation dataframe
    tokenisation_df = pd.DataFrame({
        'Language': ['Swahili', 'Yoruba', 'Amharic'],
        'Fertility_Penalty': [1.6, 2.1, 2.3],
        'OOV_Rate': [0.12, 0.18, 0.22]
    })
    
    # Dummy morphology results
    morph_results = {
        'Swahili': pd.DataFrame({'boundary_f1': [0.65, 0.58, 0.72]}),
        'Yoruba': pd.DataFrame({'boundary_f1': [0.45, 0.52, 0.48]}),
        'Amharic': pd.DataFrame({'boundary_f1': [0.55, 0.48, 0.52]})
    }
    
    # Dummy embeddings and labels
    embeddings = np.random.randn(20, 768)
    language_labels = ['English'] * 5 + ['Swahili'] * 5 + ['Yoruba'] * 5 + ['Amharic'] * 5
    topic_labels = ['antenatal'] * 4 + ['labor'] * 4 + ['postnatal'] * 4 + ['mental'] * 4 + ['vaccines'] * 4
    
    # Take baseline snapshot
    baseline = tracker.take_snapshot(
        intervention_description="Baseline - Original model",
        sdi_matrix=sdi_matrix,
        performance_df=performance_df,
        tokenisation_df=tokenisation_df,
        morph_results=morph_results,
        embeddings=embeddings,
        language_labels=language_labels,
        topic_labels=topic_labels
    )
    
    print(f"Baseline Snapshot - SDI: {baseline.avg_sdi:.3f}, Flags: {baseline.flags}")
    
    # Simulated improved metrics after intervention
    improved_sdi = pd.DataFrame(
        np.array([[0, 0.22, 0.28, 0.35],
                  [0.22, 0, 0.20, 0.30],
                  [0.28, 0.20, 0, 0.22],
                  [0.35, 0.30, 0.22, 0]]),
        index=languages, columns=languages
    )
    
    improved_perf = pd.DataFrame({
        'model_name': ['mBERT', 'mBERT', 'mBERT'],
        'language': ['English', 'Swahili', 'Yoruba'],
        'f1_disparity': [0, 0.15, 0.22],
        'exact_match': [0.87, 0.73, 0.68]
    })
    
    improved_tok = pd.DataFrame({
        'Language': ['Swahili', 'Yoruba', 'Amharic'],
        'Fertility_Penalty': [1.35, 1.65, 1.80],
        'OOV_Rate': [0.08, 0.12, 0.15]
    })
    
    improved_morph = {
        'Swahili': pd.DataFrame({'boundary_f1': [0.78, 0.72, 0.82]}),
        'Yoruba': pd.DataFrame({'boundary_f1': [0.62, 0.68, 0.65]}),
        'Amharic': pd.DataFrame({'boundary_f1': [0.68, 0.62, 0.70]})
    }
    
    # Take improved snapshot
    improved = tracker.take_snapshot(
        intervention_description="After MorphBPE + LAFT",
        sdi_matrix=improved_sdi,
        performance_df=improved_perf,
        tokenisation_df=improved_tok,
        morph_results=improved_morph,
        embeddings=embeddings * 0.8,  # Simulated improved embeddings
        language_labels=language_labels,
        topic_labels=topic_labels
    )
    
    # Register intervention
    result = tracker.register_intervention(
        intervention_name="MorphBPE + LAFT",
        before_snapshot=baseline,
        after_snapshot=improved
    )
    
    print(f"\nIntervention: {result.intervention_name}")
    print(f"Success: {result.success}")
    print(f"Improvements: {result.improvement_scores}")
    print(f"Recommendations: {result.recommendations}")
    
    # Compare snapshots
    comparison = tracker.compare_snapshots(baseline, improved)
    print("\nSnapshot Comparison:")
    print(comparison)
    
    # Generate progress report
    report = tracker.generate_progress_report()
    print("\nProgress Report:")
    print(f"Trajectory: {report['trajectory']}")
    print(f"Overall Improvements: {report['improvements']}")
    print(f"Recommendations: {report['recommendations']}")
    
    # Visualize progress
    tracker.visualize_progress(save_path=f"{self.output_dir}/bias_progress.png")
