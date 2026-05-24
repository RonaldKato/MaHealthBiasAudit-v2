"""
Enhanced Visualization Dashboard for MaHealthBiasAudit v2
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Wedge
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

from config import FIGURES_DIR, PRIMARY_LANGUAGES, THRESHOLDS

# Set style for professional visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)


class BiasVisualizationDashboard:
    """
    Enhanced visualization dashboard with unique, data-specific insights
    """
    
    def __init__(self, save_figures: bool = True, output_dir: str = None, show_display: bool = False):
        self.save_figures = save_figures
        self.show_display = show_display
        self.output_dir = output_dir or FIGURES_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.generated_figures = []
        self.bias_patterns = None
        
        # Define language-specific colors for consistency
        self.lang_colors = {
            'English': '#2E86AB',      # Blue
            'Swahili': '#A23B72',      # Purple
            'Yoruba': '#F18F01',       # Orange
            'Amharic': '#C73E1D',      # Red
            'Luganda': '#3D5A80',      # Navy
            'Runyankore': '#EE6C4D'    # Coral
        }
        
        # Define topic colors
        self.topic_colors = {
            'Nutrition': '#2ECC71',
            'Labor & Delivery': '#E74C3C',
            'Postnatal Care': '#3498DB',
            'Mental Health': '#F39C12',
            'Child Health': '#9B59B6'
        }
        
        print(f"Enhanced Visualization Dashboard initialized")
        print(f"   Output directory: {self.output_dir}")
    
    def _save_and_show(self, fig, filename: str, dpi: int = 200):
        """Helper to save and/or show figure"""
        if self.save_figures:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"   ✓ Saved: {filepath}")
        
        if self.show_display:
            plt.figure(fig.number)
            plt.show()
        
        self.generated_figures.append({'fig': fig, 'filename': filename})
        plt.close(fig)
    
    # ========================================================================
    # ORIGINAL METHODS (called in main.py)
    # ========================================================================
    
    def plot_sdi_heatmap(self, sdi_matrix: pd.DataFrame, title: str = "Semantic Divergence Index (SDI) Heatmap") -> plt.Figure:
        """Create heatmap of Semantic Divergence Index between language pairs"""
        print("\n   Creating SDI heatmap...")
        
        if sdi_matrix is None or sdi_matrix.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(sdi_matrix.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Semantic Divergence Index (higher = more bias)', fontsize=11)
        
        # Set labels
        ax.set_xticks(range(len(sdi_matrix.columns)))
        ax.set_yticks(range(len(sdi_matrix.index)))
        ax.set_xticklabels(sdi_matrix.columns, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(sdi_matrix.index, fontsize=10)
        
        # Add text annotations
        for i in range(len(sdi_matrix.index)):
            for j in range(len(sdi_matrix.columns)):
                value = sdi_matrix.iloc[i, j]
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.3f}',
                       ha="center", va="center", color=text_color, fontsize=9, fontweight='bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Language', fontsize=12)
        ax.set_ylabel('Source Language', fontsize=12)
        
        self._save_and_show(fig, "sdi_heatmap.png")
        return fig
    
    def plot_tokenisation_parity(self, tp_df: pd.DataFrame) -> plt.Figure:
        """Create visualization of tokenisation parity across languages"""
        print("\n   Creating tokenisation parity visualization...")
        
        if tp_df is None or tp_df.empty:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Fertility Penalty by Language
        fert_df = tp_df.groupby('Language')['Fertility_Penalty'].first().reset_index()
        colors = ['#E74C3C' if f > THRESHOLDS['tokenisation_parity'] else '#2ECC71' for f in fert_df['Fertility_Penalty']]
        bars = axes[0].bar(fert_df['Language'], fert_df['Fertility_Penalty'], color=colors, edgecolor='black')
        axes[0].axhline(y=THRESHOLDS['tokenisation_parity'], color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[0].set_title('Fertility Penalty by Language\n(higher = more tokens needed)', fontweight='bold')
        axes[0].set_ylabel('Fertility Penalty')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].legend()
        
        for bar, val in zip(bars, fert_df['Fertility_Penalty']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: OOV Rate
        oov_df = tp_df.groupby('Language')['OOV_Rate'].first().reset_index()
        bars = axes[1].bar(oov_df['Language'], oov_df['OOV_Rate'] * 100, color='#F39C12', edgecolor='black')
        axes[1].axhline(y=THRESHOLDS['oov_rate'] * 100, color='red', linestyle='--', linewidth=2, label='Threshold')
        axes[1].set_title('Out-of-Vocabulary Rate\n(unknown tokens)', fontweight='bold')
        axes[1].set_ylabel('OOV Rate (%)')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()
        
        for bar, val in zip(bars, oov_df['OOV_Rate'] * 100):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Tokenisation Parity Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "tokenisation_parity.png")
        return fig
    
    def plot_trust_aware_results(self, trust_results: Dict) -> plt.Figure:
        """Create visualization of Trust-Aware Module results"""
        print("\n   Creating Trust-Aware Module visualization...")
        
        if not trust_results:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        languages = list(trust_results.keys())
        trust_scores = [trust_results[l].trust_score for l in languages]
        preservation_needed = [trust_results[l].preservation_needed for l in languages]
        
        colors = ['#2ECC71' if p else '#F39C12' for p in preservation_needed]
        bars = axes[0].bar(languages, trust_scores, color=colors, edgecolor='black')
        axes[0].axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='High Trust Target')
        axes[0].set_title('Trust-Aware Analysis\n(higher = more culturally appropriate)', fontweight='bold')
        axes[0].set_ylabel('Trust Score')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, trust_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        cultural_counts = [len(trust_results[l].cultural_terms_found) for l in languages]
        bars = axes[1].bar(languages, cultural_counts, color='#9B59B6', edgecolor='black')
        axes[1].set_title('Cultural Terminology Detected\n(terms to preserve)', fontweight='bold')
        axes[1].set_ylabel('Number of Cultural Terms')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, cultural_counts):
            if count > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                            str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Trust-Aware Module: Cultural Knowledge Preservation', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "trust_aware_results.png")
        return fig
    
    def plot_rca_summary(self, rca_results: List) -> plt.Figure:
        """Create summary visualization of Root Cause Attribution results"""
        print("\n   Creating RCA summary visualization...")
        
        if not rca_results:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Root cause distribution
        cause_counts = {}
        preserve_counts = 0
        for result in rca_results:
            cause_counts[result.root_cause] = cause_counts.get(result.root_cause, 0) + 1
            if result.preserve:
                preserve_counts += 1
        
        causes = list(cause_counts.keys())
        counts = list(cause_counts.values())
        colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C', '#9B59B6']
        
        bars = axes[0].bar(causes, counts, color=colors[:len(causes)], edgecolor='black')
        axes[0].set_title('Root Cause Attribution\n(primary sources of bias)', fontweight='bold')
        axes[0].set_ylabel('Number of Cases')
        axes[0].tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Preservation vs Intervention
        intervention = len(rca_results) - preserve_counts
        sizes = [preserve_counts, intervention]
        labels = ['Preserve (Cultural)', 'Intervention Needed']
        colors_pie = ['#2ECC71', '#E74C3C']
        
        axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                   startangle=90, explode=(0.05, 0), shadow=True)
        axes[1].set_title('Bias Resolution Strategy\n(Cultural vs Technical)', fontweight='bold')
        
        plt.suptitle('Root Cause Attribution (RCA) Cascade Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "rca_summary.png")
        return fig
    
    def plot_bias_pattern_heatmap(self, patterns_df: pd.DataFrame) -> plt.Figure:
        """Create heatmap of bias patterns by topic"""
        print("\n   Creating bias patterns heatmap...")
        
        if patterns_df is None or patterns_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        topics = patterns_df['topic'].tolist()
        sdi_values = patterns_df['avg_sdi'].tolist()
        severities = patterns_df['bias_severity'].tolist()
        
        color_map = {'low': '#2ECC71', 'moderate': '#F39C12', 'high': '#E74C3C'}
        bar_colors = [color_map.get(s, '#3498DB') for s in severities]
        
        y_pos = np.arange(len(topics))
        bars = ax.barh(y_pos, sdi_values, color=bar_colors, edgecolor='black', height=0.6)
        
        ax.axvline(x=THRESHOLDS['sdi_moderate'], color='#F39C12', linestyle='--', linewidth=2, alpha=0.7, label='Moderate (0.2)')
        ax.axvline(x=THRESHOLDS['sdi_high'], color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7, label='High (0.4)')
        
        for bar, value, severity in zip(bars, sdi_values, severities):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f} ({severity})', ha='left', va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t.replace('_', ' ').title() for t in topics])
        ax.set_xlabel('Semantic Divergence Index (SDI)', fontsize=12)
        ax.set_title('Bias Patterns by Maternal Health Topic\n(higher SDI = more bias)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, max(sdi_values) + 0.1)
        
        self._save_and_show(fig, "bias_patterns.png")
        return fig
    
    def plot_performance_comparison(self, performance_df: pd.DataFrame) -> plt.Figure:
        """Create model performance comparison visualization"""
        print("\n   Creating performance comparison visualization...")
        
        if performance_df is None or performance_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        perf_by_lang = performance_df.groupby('language')['token_f1'].mean().reset_index()
        bars = ax.bar(perf_by_lang['language'], perf_by_lang['token_f1'], 
                     color='#3498DB', edgecolor='black')
        ax.axhline(y=0.7, color='#2ECC71', linestyle='--', alpha=0.7, label='Target (0.7)')
        ax.set_title('Average Model Performance\n(F1 Score by Language)', fontweight='bold')
        ax.set_ylabel('Token F1')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1)
        ax.legend()
        
        for bar, val in zip(bars, perf_by_lang['token_f1']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        self._save_and_show(fig, "performance_comparison.png")
        return fig
    
    def plot_3d_embedding_space(self, embeddings: np.ndarray, 
                                 language_labels: List[str], 
                                 topic_labels: List[str] = None,
                                 title: str = "Multilingual Embedding Space") -> plt.Figure:
        """Create 3D t-SNE visualization of embedding space"""
        print("\n   Creating 3D embedding space visualization...")
        
        if len(embeddings) == 0:
            return None
        
        # Reduce to 3D using t-SNE
        n_samples = len(embeddings)
        perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
        
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        embeddings_3d = tsne.fit_transform(embeddings)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        unique_langs = list(set(language_labels))
        
        for lang in unique_langs:
            mask = [l == lang for l in language_labels]
            if any(mask):
                color = self.lang_colors.get(lang, '#3498DB')
                ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                          label=lang, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, color=color)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_zlabel('t-SNE Dimension 3', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        self._save_and_show(fig, "3d_embedding_space.png")
        return fig
    
    def plot_interrogative_analysis(self, inter_df: pd.DataFrame) -> plt.Figure:
        """Create visualization of interrogative structure analysis"""
        print("\n   Creating interrogative structure analysis...")
        
        if inter_df is None or inter_df.empty:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Interrogative type distribution
        type_counts = pd.crosstab(inter_df['Language'], inter_df['Actual_Type'])
        type_counts.plot(kind='bar', ax=axes[0], edgecolor='black')
        axes[0].set_title('Interrogative Structure by Language', fontweight='bold')
        axes[0].set_xlabel('Language')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Mismatch detection
        mismatch_by_lang = inter_df.groupby('Language')['Mismatch'].sum()
        if not mismatch_by_lang.empty:
            colors = ['#E74C3C' if v > 0 else '#2ECC71' for v in mismatch_by_lang.values]
            bars = axes[1].bar(mismatch_by_lang.index, mismatch_by_lang.values, color=colors, edgecolor='black')
            axes[1].set_title('Query Structure Mismatch\n(Non-English patterns)', fontweight='bold')
            axes[1].set_ylabel('Number of Mismatches')
            axes[1].set_xlabel('Language')
            axes[1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, mismatch_by_lang.values):
                if value > 0:
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                                str(int(value)), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Interrogative Structure Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "interrogative_analysis.png")
        return fig
    
    def create_comprehensive_dashboard(self, sdi_matrix: pd.DataFrame, tp_df: pd.DataFrame,
                                        performance_df: pd.DataFrame, trust_results: Dict,
                                        rca_results: List, bias_patterns: pd.DataFrame) -> plt.Figure:
        """Create comprehensive dashboard"""
        print("\n   Creating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. SDI Heatmap (simplified for dashboard)
        ax1 = fig.add_subplot(gs[0, 0])
        if sdi_matrix is not None and not sdi_matrix.empty:
            im = ax1.imshow(sdi_matrix.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(len(sdi_matrix.columns)))
            ax1.set_yticks(range(len(sdi_matrix.index)))
            ax1.set_xticklabels(sdi_matrix.columns, rotation=45, ha='right', fontsize=8)
            ax1.set_yticklabels(sdi_matrix.index, fontsize=8)
            ax1.set_title('SDI Heatmap', fontweight='bold', fontsize=10)
            plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # 2. Fertility Penalty
        ax2 = fig.add_subplot(gs[0, 1])
        if tp_df is not None and not tp_df.empty:
            fert_df = tp_df.groupby('Language')['Fertility_Penalty'].first().reset_index()
            colors = ['#E74C3C' if f > THRESHOLDS['tokenisation_parity'] else '#2ECC71' for f in fert_df['Fertility_Penalty']]
            ax2.bar(fert_df['Language'], fert_df['Fertility_Penalty'], color=colors, edgecolor='black')
            ax2.axhline(y=THRESHOLDS['tokenisation_parity'], color='red', linestyle='--', label='Threshold')
            ax2.set_title('Fertility Penalty', fontweight='bold', fontsize=10)
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(fontsize=8)
        
        # 3. Trust Scores
        ax3 = fig.add_subplot(gs[0, 2])
        if trust_results:
            langs = list(trust_results.keys())
            scores = [trust_results[l].trust_score for l in langs]
            colors = ['#2ECC71' if s > 0.7 else '#F39C12' if s > 0.5 else '#E74C3C' for s in scores]
            ax3.bar(langs, scores, color=colors, edgecolor='black')
            ax3.axhline(y=0.7, color='green', linestyle='--', label='Target')
            ax3.set_title('Trust Score', fontweight='bold', fontsize=10)
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend(fontsize=8)
        
        # 4. Bias by Topic
        ax4 = fig.add_subplot(gs[1, 0])
        if bias_patterns is not None and not bias_patterns.empty:
            topics = bias_patterns['topic'].tolist()[:4]
            sdi_vals = bias_patterns['avg_sdi'].tolist()[:4]
            colors = ['#E74C3C' if v > 0.4 else '#F39C12' if v > 0.2 else '#2ECC71' for v in sdi_vals]
            y_pos = np.arange(len(topics))
            ax4.barh(y_pos, sdi_vals, color=colors, edgecolor='black')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([t.replace('_', ' ').title() for t in topics], fontsize=9)
            ax4.set_xlabel('SDI')
            ax4.set_title('Top Bias Topics', fontweight='bold', fontsize=10)
        
        # 5. RCA Summary
        ax5 = fig.add_subplot(gs[1, 1])
        if rca_results:
            cause_counts = {}
            for result in rca_results:
                cause_counts[result.root_cause] = cause_counts.get(result.root_cause, 0) + 1
            causes = list(cause_counts.keys())
            counts = list(cause_counts.values())
            colors_pie = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
            ax5.pie(counts, labels=causes, colors=colors_pie[:len(causes)], autopct='%1.0f%%', startangle=90)
            ax5.set_title('RCA: Bias Sources', fontweight='bold', fontsize=10)
        
        # 6. Performance Summary
        ax6 = fig.add_subplot(gs[1, 2])
        if performance_df is not None and not performance_df.empty:
            perf_by_lang = performance_df.groupby('language')['token_f1'].mean().reset_index()
            ax6.bar(perf_by_lang['language'], perf_by_lang['token_f1'], color='#3498DB', edgecolor='black')
            ax6.axhline(y=0.7, color='green', linestyle='--', label='Target')
            ax6.set_title('Model Performance', fontweight='bold', fontsize=10)
            ax6.set_ylim(0, 1)
            ax6.tick_params(axis='x', rotation=45)
        
        # 7. Summary Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        avg_sdi = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)].mean() if sdi_matrix is not None else 0
        
        summary_data = [
            ['Metric', 'Value', 'Status'],
            ['Average SDI', f'{avg_sdi:.3f}', '🔴 High' if avg_sdi > 0.4 else '🟡 Moderate' if avg_sdi > 0.2 else '🟢 Low'],
            ['Cultural Terms', str(sum(1 for r in rca_results if r.preserve) if rca_results else 0), 'Preserve'],
            ['RCA Cases', str(len(rca_results) if rca_results else 0), 'Action Needed'],
        ]
        
        table = ax7.table(cellText=summary_data, loc='center', cellLoc='left', colWidths=[0.35, 0.3, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax7.set_title('Executive Summary', fontweight='bold', fontsize=12)
        
        plt.suptitle('MaHealthBiasAudit v2 - Comprehensive Bias Dashboard', fontsize=16, fontweight='bold')
        self._save_and_show(fig, "comprehensive_dashboard.png", dpi=150)
        
        return fig
    
    # ========================================================================
    # ENHANCED METHODS (additional unique visualizations)
    # ========================================================================
    
    def plot_maternal_health_bias_radar(self, sdi_matrix: pd.DataFrame) -> plt.Figure:
        """Radar chart showing bias magnitude per language pair"""
        print("\n Creating Maternal Health Bias Radar Chart...")
        
        if sdi_matrix is None or sdi_matrix.empty:
            return None
        
        languages = list(sdi_matrix.columns)
        n_langs = len(languages)
        
        avg_bias = []
        for lang in languages:
            others = [l for l in languages if l != lang]
            bias_values = [sdi_matrix.loc[lang, other] for other in others]
            avg_bias.append(np.mean(bias_values))
        
        # FIX: Both arrays must have same length
        angles = np.linspace(0, 2 * np.pi, n_langs, endpoint=False).tolist()
        
        # Close the loop by appending first value to both arrays
        angles_closed = angles + angles[:1]
        avg_bias_closed = avg_bias + avg_bias[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles_closed, avg_bias_closed, 'o-', linewidth=3, color='#2C3E50', markersize=10)
        ax.fill(angles_closed, avg_bias_closed, alpha=0.25, color='#3498DB')
        
        ax.set_ylim(0, 0.7)
        ax.set_yticks([0.2, 0.4, 0.6])
        ax.set_yticklabels(['Low (0.2)', 'Moderate (0.4)', 'High (0.6)'], fontsize=9)
        
        # Add threshold circles
        theta = np.linspace(0, 2 * np.pi, 100)
        for threshold, color, label in [(0.2, '#F39C12', 'Moderate'), (0.4, '#E74C3C', 'High')]:
            r = np.full_like(theta, threshold)
            ax.plot(theta, r, '--', color=color, linewidth=1.5, alpha=0.5, label=f'{label} Bias Threshold')
        
        ax.set_xticks(angles)
        ax.set_xticklabels(languages, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1), fontsize=9)
        ax.set_title('Maternal Health Bias Magnitude by Language\n(Larger area = Higher Cross-Lingual Bias)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add value annotations
        for i, (angle, bias) in enumerate(zip(angles, avg_bias)):
            ax.annotate(f'{bias:.3f}', 
                       xy=(angle, bias + 0.03),
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        self._save_and_show(fig, "01_maternal_health_bias_radar.png")
        return fig
    
    def plot_semantic_divergence_network(self, sdi_matrix: pd.DataFrame) -> plt.Figure:
        """Network graph showing semantic divergence connections"""
        print("\n Creating Semantic Divergence Network...")
        
        if sdi_matrix is None or sdi_matrix.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        languages = list(sdi_matrix.columns)
        n = len(languages)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        positions = {lang: (np.cos(angle) * 2, np.sin(angle) * 2) for lang, angle in zip(languages, angles)}
        
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i < j:
                    sdi = sdi_matrix.loc[lang1, lang2]
                    width = 1 + (sdi / 0.6) * 7
                    width = max(1, min(8, width))
                    color = '#E74C3C' if sdi > 0.4 else '#F39C12' if sdi > 0.2 else '#2ECC71'
                    
                    x1, y1 = positions[lang1]
                    x2, y2 = positions[lang2]
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.7)
                    
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.annotate(f'{sdi:.2f}', xy=(mid_x, mid_y), ha='center', va='center',
                               fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        for lang, (x, y) in positions.items():
            # Node size based on resource level
            resource_sizes = {'high': 0.28, 'medium': 0.23, 'low': 0.18, 'very_low': 0.15}
            size = resource_sizes.get('medium', 0.2)
            
            circle = Circle((x, y), radius=size, facecolor=self.lang_colors.get(lang, '#3498DB'), 
                           edgecolor='white', linewidth=2.5, alpha=0.9)
            ax.add_patch(circle)
            ax.annotate(lang, xy=(x, y), ha='center', va='center', fontsize=11, fontweight='bold', color='white')
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='#2ECC71', linewidth=3, label='Low Bias (SDI < 0.2)'),
            Line2D([0], [0], color='#F39C12', linewidth=3, label='Moderate Bias (0.2-0.4)'),
            Line2D([0], [0], color='#E74C3C', linewidth=3, label='High Bias (SDI > 0.4)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        
        ax.set_xlim(-2.8, 2.8)
        ax.set_ylim(-2.8, 2.8)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Semantic Divergence Network: Cross-Lingual Bias in Maternal Health\n(Thicker/Redder Lines = Higher Bias)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        self._save_and_show(fig, "02_semantic_divergence_network.png")
        return fig
    
    def create_all_visualizations(self, sdi_matrix: pd.DataFrame, tp_df: pd.DataFrame,
                                   trust_results: Dict, rca_results: List,
                                   bias_patterns: pd.DataFrame, inter_df: pd.DataFrame,
                                   embeddings: np.ndarray, labels: List[str]) -> Dict:
        """Create all enhanced visualizations"""
        print("\n" + "="*60)
        print("Creating Enhanced Visualizations")
        print("="*60)
        
        visualizations = {}
        
        # Standard visualizations
        if sdi_matrix is not None and not sdi_matrix.empty:
            visualizations['sdi_heatmap'] = self.plot_sdi_heatmap(sdi_matrix)
            visualizations['radar'] = self.plot_maternal_health_bias_radar(sdi_matrix)
            visualizations['network'] = self.plot_semantic_divergence_network(sdi_matrix)
        
        if tp_df is not None and not tp_df.empty:
            visualizations['tokenisation'] = self.plot_tokenisation_parity(tp_df)
        
        if trust_results:
            visualizations['trust'] = self.plot_trust_aware_results(trust_results)
        
        if rca_results:
            visualizations['rca'] = self.plot_rca_summary(rca_results)
        
        if bias_patterns is not None and not bias_patterns.empty:
            visualizations['patterns'] = self.plot_bias_pattern_heatmap(bias_patterns)
        
        if inter_df is not None and not inter_df.empty:
            visualizations['interrogative'] = self.plot_interrogative_analysis(inter_df)
        
        if embeddings is not None and len(embeddings) > 0 and labels:
            visualizations['embedding_3d'] = self.plot_3d_embedding_space(embeddings, labels)
        
        # Comprehensive dashboard
        performance_df = None  # This would come from model results
        visualizations['dashboard'] = self.create_comprehensive_dashboard(
            sdi_matrix, tp_df, performance_df, trust_results, rca_results, bias_patterns
        )
        
        print(f"\n Created {len(visualizations)} visualizations")
        print(f"   Output directory: {self.output_dir}")
        
        return visualizations


# Import LANGUAGES at the end to avoid circular import
from config import LANGUAGES