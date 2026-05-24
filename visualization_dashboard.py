"""
Visualization Dashboard for MaHealthBiasAudit v2
Creates comprehensive visualizations for bias audit reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from config import FIGURES_DIR, PRIMARY_LANGUAGES, THRESHOLDS, VIZ_SETTINGS

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.2)


class BiasVisualizationDashboard:
    """
    Handles all visualization creation for the bias audit pipeline
    Automatically saves figures to output directory
    """
    
    def __init__(self, save_figures: bool = True, output_dir: str = None, show_display: bool = False):
        """
        Initialize the visualization dashboard
        
        Args:
            save_figures: Whether to save figures to disk
            output_dir: Directory to save figures
            show_display: Whether to show figures interactively
        """
        self.save_figures = save_figures
        self.show_display = show_display
        
        if output_dir is None:
            output_dir = FIGURES_DIR
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Store generated figures
        self.generated_figures = []
        self.bias_patterns = None
        
        print(f"✅ Visualization Dashboard initialized")
        print(f"   Save figures: {save_figures} → {output_dir}")
    
    def _save_and_show(self, fig, filename: str, dpi: int = 150):
        """Helper to save and/or show figure"""
        if self.save_figures:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"   ✓ Saved: {filepath}")
        
        if self.show_display:
            plt.figure(fig.number)
            plt.show()
        
        self.generated_figures.append({'fig': fig, 'filename': filename})
        plt.close(fig)
    
    def plot_sdi_heatmap(self, sdi_matrix: pd.DataFrame, 
                         title: str = "Semantic Divergence Index (SDI) Heatmap") -> plt.Figure:
        """
        Create heatmap of Semantic Divergence Index between language pairs
        """
        print("\n   Creating SDI heatmap...")
        
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
        
        # Add threshold lines
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(x=0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Language', fontsize=12)
        ax.set_ylabel('Source Language', fontsize=12)
        
        # Add interpretation text
        upper_tri = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)]
        avg_sdi = np.mean(upper_tri) if len(upper_tri) > 0 else 0
        fig.text(0.02, 0.02, f"Average SDI: {avg_sdi:.3f} | Thresholds: >0.2=moderate, >0.4=high", 
                fontsize=10, style='italic', color='gray')
        
        self._save_and_show(fig, "sdi_heatmap.png")
        return fig
    
    def plot_tokenisation_parity(self, tp_df: pd.DataFrame) -> plt.Figure:
        """
        Create visualization of tokenisation parity across languages
        """
        print("\n   Creating tokenisation parity visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Fertility Penalty by Tokeniser
        if 'Fertility_Penalty' in tp_df.columns and 'Tokeniser' in tp_df.columns:
            # Pivot for grouped bar chart
            pivot_df = tp_df.pivot(index='Language', columns='Tokeniser', values='Fertility_Penalty')
            
            languages = pivot_df.index.tolist()
            x = np.arange(len(languages))
            width = 0.25
            
            colors = {'mBERT': 'skyblue', 'XLM-R': 'lightgreen', 'AfriBERTa': 'coral'}
            for i, tokeniser in enumerate(pivot_df.columns):
                axes[0].bar(x + (i - 1) * width, pivot_df[tokeniser], width, 
                           label=tokeniser, color=colors.get(tokeniser, 'gray'))
            
            axes[0].axhline(y=THRESHOLDS['tokenisation_parity'], color='red', 
                          linestyle='--', linewidth=2, label='Threshold')
            axes[0].set_title('Fertility Penalty by Tokeniser\n(higher = more tokens needed)', fontweight='bold')
            axes[0].set_ylabel('Fertility Penalty')
            axes[0].set_xlabel('Language')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(languages, rotation=45, ha='right')
            axes[0].legend()
            axes[0].set_ylim(0, 3)
        
        # Plot 2: OOV Rate
        if 'OOV_Rate' in tp_df.columns:
            # Get unique languages (take first occurrence)
            oov_df = tp_df.groupby('Language')['OOV_Rate'].first().reset_index()
            bars = axes[1].bar(oov_df['Language'], oov_df['OOV_Rate'] * 100, 
                              color='lightcoral', edgecolor='black')
            axes[1].axhline(y=THRESHOLDS['oov_rate'] * 100, color='red', 
                          linestyle='--', linewidth=2, label='Threshold')
            axes[1].set_title('Out-of-Vocabulary Rate\n(unknown tokens)', fontweight='bold')
            axes[1].set_ylabel('OOV Rate (%)')
            axes[1].set_xlabel('Language')
            axes[1].legend()
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, oov_df['OOV_Rate'] * 100):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Best Tokeniser
        if 'Best_Tokeniser' in tp_df.columns:
            best_df = tp_df.groupby('Language')['Best_Tokeniser'].first().reset_index()
            best_counts = best_df['Best_Tokeniser'].value_counts()
            colors = {'mBERT': 'skyblue', 'XLM-R': 'lightgreen', 'AfriBERTa': 'coral'}
            bar_colors = [colors.get(t, 'gray') for t in best_counts.index]
            
            bars = axes[2].bar(best_counts.index, best_counts.values, color=bar_colors, edgecolor='black')
            axes[2].set_title('Best Tokeniser per Language\n(lowest fertility penalty)', fontweight='bold')
            axes[2].set_ylabel('Number of Languages')
            axes[2].set_xlabel('Tokeniser')
            
            for bar, value in zip(bars, best_counts.values):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Tokenisation Parity Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_and_show(fig, "tokenisation_parity.png")
        
        return fig
    
    def plot_3d_embedding_space(self, embeddings: np.ndarray, 
                                 language_labels: List[str], 
                                 topic_labels: List[str] = None,
                                 title: str = "Multilingual Embedding Space") -> plt.Figure:
        """
        Create 3D t-SNE visualization of embedding space
        """
        print("\n   Creating 3D embedding space visualization...")
        
        # Reduce to 3D using t-SNE
        n_samples = len(embeddings)
        perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
        
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        embeddings_3d = tsne.fit_transform(embeddings)
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by language
        unique_langs = list(set(language_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_langs)))
        color_map = {lang: colors[i] for i, lang in enumerate(unique_langs)}
        
        for lang in unique_langs:
            mask = [l == lang for l in language_labels]
            if any(mask):
                ax.scatter(embeddings_3d[mask, 0], 
                          embeddings_3d[mask, 1], 
                          embeddings_3d[mask, 2],
                          c=[color_map[lang]], 
                          label=lang, 
                          s=80, 
                          alpha=0.7,
                          edgecolors='black',
                          linewidth=0.5)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12, labelpad=10)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12, labelpad=10)
        ax.set_zlabel('t-SNE Dimension 3', fontsize=12, labelpad=10)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
        
        # Add note about interpretation
        fig.text(0.02, 0.02, "Interpretation: Clusters by language = bias detected", 
                fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        self._save_and_show(fig, "3d_embedding_space.png")
        
        return fig
    
    def plot_bias_pattern_heatmap(self, patterns_df: pd.DataFrame) -> plt.Figure:
        """
        Create heatmap of bias patterns by topic
        """
        print("\n   Creating bias patterns heatmap...")
        
        if patterns_df is None or patterns_df.empty:
            print("     No bias pattern data to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        topics = patterns_df['topic'].tolist()
        sdi_values = patterns_df['avg_sdi'].tolist()
        severities = patterns_df['bias_severity'].tolist()
        
        # Color mapping
        color_map = {'low': 'green', 'moderate': 'orange', 'high': 'red'}
        bar_colors = [color_map.get(s, 'blue') for s in severities]
        
        # Horizontal bar chart
        y_pos = np.arange(len(topics))
        bars = ax.barh(y_pos, sdi_values, color=bar_colors, edgecolor='black', height=0.6)
        
        # Add threshold lines
        ax.axvline(x=THRESHOLDS['sdi_moderate'], color='orange', linestyle='--', 
                  linewidth=2, alpha=0.7, label='Moderate (0.2)')
        ax.axvline(x=THRESHOLDS['sdi_high'], color='red', linestyle='--', 
                  linewidth=2, alpha=0.7, label='High (0.4)')
        
        # Add value labels
        for bar, value, severity in zip(bars, sdi_values, severities):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f} ({severity})', ha='left', va='center', fontsize=9)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([t.replace('_', ' ').title() for t in topics])
        ax.set_xlabel('Semantic Divergence Index (SDI)', fontsize=12)
        ax.set_title('Bias Patterns by Maternal Health Topic\n(higher SDI = more bias)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
        
        self._save_and_show(fig, "bias_patterns.png")
        return fig
    
    def plot_interrogative_analysis(self, inter_df: pd.DataFrame) -> plt.Figure:
        """
        Create visualization of interrogative structure analysis
        """
        print("\n   Creating interrogative structure analysis...")
        
        if inter_df is None or inter_df.empty:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Interrogative type distribution by language
        type_counts = pd.crosstab(inter_df['Language'], inter_df['Actual_Type'])
        type_counts.plot(kind='bar', ax=axes[0], edgecolor='black')
        axes[0].set_title('Interrogative Structure by Language\n(wh-fronted vs inline vs sentence-final)', 
                         fontweight='bold')
        axes[0].set_xlabel('Language')
        axes[0].set_ylabel('Count')
        axes[0].legend(title='Question Type')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Mismatch detection
        mismatch_by_lang = inter_df.groupby('Language')['Mismatch'].sum()
        if not mismatch_by_lang.empty:
            colors = ['red' if v > 0 else 'green' for v in mismatch_by_lang.values]
            bars = axes[1].bar(mismatch_by_lang.index, mismatch_by_lang.values, color=colors, edgecolor='black')
            axes[1].set_title('Query Structure Mismatch\n(English vs Non-English patterns)', fontweight='bold')
            axes[1].set_ylabel('Number of Mismatches')
            axes[1].set_xlabel('Language')
            axes[1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, mismatch_by_lang.values):
                if value > 0:
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               str(int(value)), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Interrogative Structure Analysis\n(Source of Query Formulation Bias)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "interrogative_analysis.png")
        
        return fig
    
    def plot_trust_aware_results(self, trust_results: Dict) -> plt.Figure:
        """
        Create visualization of Trust-Aware Module results
        """
        print("\n   Creating Trust-Aware Module visualization...")
        
        if not trust_results:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Trust scores by language
        languages = list(trust_results.keys())
        trust_scores = [trust_results[l].trust_score for l in languages]
        preservation_needed = [trust_results[l].preservation_needed for l in languages]
        
        colors = ['coral' if p else 'skyblue' for p in preservation_needed]
        bars = axes[0].bar(languages, trust_scores, color=colors, edgecolor='black')
        axes[0].axhline(y=THRESHOLDS['trust_score_target'], color='green', linestyle='--', 
                       linewidth=2, label='High Trust Target')
        axes[0].set_title('Trust-Aware Analysis\n(higher = more culturally appropriate)', fontweight='bold')
        axes[0].set_ylabel('Trust Score')
        axes[0].set_xlabel('Language')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        for bar, score in zip(bars, trust_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Cultural terms found
        cultural_counts = [len(trust_results[l].cultural_terms_found) for l in languages]
        bars = axes[1].bar(languages, cultural_counts, color='lightgreen', edgecolor='black')
        axes[1].set_title('Cultural Terminology Detected\n(terms to preserve)', fontweight='bold')
        axes[1].set_ylabel('Number of Cultural Terms')
        axes[1].set_xlabel('Language')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, cultural_counts):
            if count > 0:
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Trust-Aware Module: Cultural Knowledge Preservation', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "trust_aware_results.png")
        
        return fig
    
    def plot_rca_summary(self, rca_results: List) -> plt.Figure:
        """
        Create summary visualization of Root Cause Attribution results
        """
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
        
        colors = {'TOKENISATION': 'skyblue', 'MORPHOLOGY': 'lightgreen', 
                  'QUERY_STRUCTURE': 'orange', 'CULTURAL': 'coral', 'UNKNOWN': 'gray'}
        bar_colors = [colors.get(c, 'gray') for c in causes]
        
        bars = axes[0].bar(causes, counts, color=bar_colors, edgecolor='black')
        axes[0].set_title('Root Cause Attribution\n(primary sources of bias)', fontweight='bold')
        axes[0].set_ylabel('Number of Cases')
        axes[0].set_xlabel('Root Cause')
        axes[0].tick_params(axis='x', rotation=45)
        
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Preservation vs Intervention
        intervention = len(rca_results) - preserve_counts
        sizes = [preserve_counts, intervention]
        labels = ['Preserve (Cultural)', 'Intervention Needed']
        colors_pie = ['green', 'red']
        
        axes[1].pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                   startangle=90, explode=(0.05, 0), shadow=True)
        axes[1].set_title('Bias Resolution Strategy\n(Cultural vs Technical)', fontweight='bold')
        
        plt.suptitle('Root Cause Attribution (RCA) Cascade Results', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "rca_summary.png")
        
        return fig
    
    def plot_performance_comparison(self, performance_df: pd.DataFrame) -> plt.Figure:
        """
        Create model performance comparison visualization
        """
        print("\n   Creating performance comparison visualization...")
        
        if performance_df is None or performance_df.empty:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: F1 Score by language and training condition
        pivot_f1 = performance_df.pivot_table(index='language', columns='training_condition', 
                                              values='token_f1', aggfunc='first')
        
        pivot_f1.plot(kind='bar', ax=axes[0], edgecolor='black')
        axes[0].set_title('Model Performance (Token F1) by Language\nand Training Condition', fontweight='bold')
        axes[0].set_ylabel('Token F1 Score')
        axes[0].set_xlabel('Language')
        axes[0].legend(title='Training')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Transfer gain
        if 'transfer_gain' in performance_df.columns:
            gain_df = performance_df[performance_df['transfer_gain'].notna()]
            if not gain_df.empty:
                colors = ['green' if g > 0 else 'red' for g in gain_df['transfer_gain']]
                bars = axes[1].bar(gain_df['language'], gain_df['transfer_gain'], color=colors, edgecolor='black')
                axes[1].axhline(y=0, color='black', linewidth=1)
                axes[1].set_title('Cross-Lingual Transfer Gain\n(FT-Lang vs FT-EN)', fontweight='bold')
                axes[1].set_ylabel('Transfer Gain (ΔF1)')
                axes[1].set_xlabel('Language')
                axes[1].tick_params(axis='x', rotation=45)
                
                for bar, gain in zip(bars, gain_df['transfer_gain']):
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if gain > 0 else -0.05),
                               f'{gain:.3f}', ha='center', va='bottom' if gain > 0 else 'top', fontweight='bold')
        
        plt.suptitle('Model Performance and Transfer Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "performance_comparison.png")
        
        return fig
    
    def create_comprehensive_dashboard(self, 
                                        sdi_matrix: pd.DataFrame,
                                        tp_df: pd.DataFrame,
                                        performance_df: pd.DataFrame,
                                        trust_results: Dict,
                                        rca_results: List,
                                        bias_patterns: pd.DataFrame) -> plt.Figure:
        """
        Create a comprehensive dashboard combining multiple visualizations
        """
        print("\n   Creating comprehensive dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # 1. SDI Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        if sdi_matrix is not None and not sdi_matrix.empty:
            im = ax1.imshow(sdi_matrix.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(len(sdi_matrix.columns)))
            ax1.set_yticks(range(len(sdi_matrix.index)))
            ax1.set_xticklabels(sdi_matrix.columns, rotation=45, ha='right', fontsize=8)
            ax1.set_yticklabels(sdi_matrix.index, fontsize=8)
            ax1.set_title('SDI Heatmap\n(higher = more bias)', fontweight='bold', fontsize=10)
            plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # 2. Fertility Penalty
        ax2 = fig.add_subplot(gs[0, 1])
        if tp_df is not None and not tp_df.empty:
            fert_df = tp_df.groupby('Language')['Fertility_Penalty'].first().reset_index()
            colors = ['red' if f > THRESHOLDS['tokenisation_parity'] else 
                     'orange' if f > THRESHOLDS['tokenisation_parity'] * 0.8 else 'green' 
                     for f in fert_df['Fertility_Penalty']]
            bars = ax2.bar(fert_df['Language'], fert_df['Fertility_Penalty'], color=colors, edgecolor='black')
            ax2.axhline(y=THRESHOLDS['tokenisation_parity'], color='red', linestyle='--', 
                       linewidth=2, label='Threshold')
            ax2.set_title('Fertility Penalty\n(>1.5 = significant bias)', fontweight='bold', fontsize=10)
            ax2.set_ylabel('Fertility Penalty')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(fontsize=8)
            
            for bar, val in zip(bars, fert_df['Fertility_Penalty']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Trust Scores
        ax3 = fig.add_subplot(gs[0, 2])
        if trust_results:
            languages = list(trust_results.keys())
            trust_scores = [trust_results[l].trust_score for l in languages]
            colors = ['green' if t > THRESHOLDS['trust_score_target'] else 
                     'orange' if t > 0.5 else 'red' for t in trust_scores]
            bars = ax3.bar(languages, trust_scores, color=colors, edgecolor='black')
            ax3.axhline(y=THRESHOLDS['trust_score_target'], color='green', linestyle='--', 
                       alpha=0.7, label='Target')
            ax3.set_title('Trust Score\n(cultural appropriateness)', fontweight='bold', fontsize=10)
            ax3.set_ylabel('Trust Score')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend(fontsize=8)
            for bar, val in zip(bars, trust_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 4. Bias Patterns by Topic
        ax4 = fig.add_subplot(gs[1, 0])
        if bias_patterns is not None and not bias_patterns.empty:
            topics = bias_patterns['topic'].tolist()
            sdi_values = bias_patterns['avg_sdi'].tolist()
            severities = bias_patterns['bias_severity'].tolist()
            color_map = {'low': 'green', 'moderate': 'orange', 'high': 'red'}
            bar_colors = [color_map.get(s, 'blue') for s in severities]
            
            y_pos = np.arange(len(topics))
            bars = ax4.barh(y_pos, sdi_values, color=bar_colors, edgecolor='black', height=0.6)
            ax4.axvline(x=THRESHOLDS['sdi_moderate'], color='orange', linestyle='--', 
                       alpha=0.7, label='Moderate')
            ax4.axvline(x=THRESHOLDS['sdi_high'], color='red', linestyle='--', 
                       alpha=0.7, label='High')
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels([t.replace('_', ' ').title() for t in topics])
            ax4.set_xlabel('SDI')
            ax4.set_title('Bias by Topic', fontweight='bold', fontsize=10)
            ax4.legend(fontsize=8)
            for bar, val in zip(bars, sdi_values):
                ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', ha='left', va='center', fontsize=8)
        
        # 5. RCA Summary
        ax5 = fig.add_subplot(gs[1, 1])
        if rca_results:
            cause_counts = {}
            for result in rca_results:
                cause_counts[result.root_cause] = cause_counts.get(result.root_cause, 0) + 1
            causes = list(cause_counts.keys())
            counts = list(cause_counts.values())
            colors = {'TOKENISATION': 'skyblue', 'MORPHOLOGY': 'lightgreen', 
                      'QUERY_STRUCTURE': 'orange', 'CULTURAL': 'coral', 'UNKNOWN': 'gray'}
            bar_colors = [colors.get(c, 'gray') for c in causes]
            bars = ax5.bar(causes, counts, color=bar_colors, edgecolor='black')
            ax5.set_title('RCA: Bias Sources', fontweight='bold', fontsize=10)
            ax5.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars, counts):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom', fontsize=8)
        
        # 6. Performance Summary
        ax6 = fig.add_subplot(gs[1, 2])
        if performance_df is not None and not performance_df.empty:
            # Show average F1 by language
            perf_by_lang = performance_df.groupby('language')['token_f1'].mean().reset_index()
            bars = ax6.bar(perf_by_lang['language'], perf_by_lang['token_f1'], 
                          color='skyblue', edgecolor='black')
            ax6.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Target')
            ax6.set_title('Average Model Performance\n(F1 Score by Language)', fontweight='bold', fontsize=10)
            ax6.set_ylabel('Token F1')
            ax6.tick_params(axis='x', rotation=45)
            ax6.set_ylim(0, 1)
            ax6.legend(fontsize=8)
            for bar, val in zip(bars, perf_by_lang['token_f1']):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 7. Summary Table
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        # Calculate summary metrics
        avg_sdi = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)].mean() if sdi_matrix is not None else 0
        lang_purity = "N/A"
        
        summary_data = [
            ['Metric', 'Value', 'Status'],
            ['Average SDI', f'{avg_sdi:.3f}', '🔴 High' if avg_sdi > 0.4 else '🟡 Moderate' if avg_sdi > 0.2 else '🟢 Low'],
            ['Cultural Terms to Preserve', str(sum(1 for r in rca_results if r.preserve) if rca_results else 0), '✅ Document'],
            ['RCA Cases Analyzed', str(len(rca_results) if rca_results else 0), 'Action Needed' if rca_results else 'None'],
            ['Languages with High Fertility', str(len(tp_df[tp_df['Fertility_Penalty'] > THRESHOLDS['tokenisation_parity']]['Language'].unique()) if tp_df is not None else 0), '⚠️ Needs Optimization']
        ]
        
        table = ax7.table(cellText=summary_data, loc='center', cellLoc='left', 
                         colWidths=[0.35, 0.3, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax7.set_title('Executive Summary', fontweight='bold', fontsize=12)
        
        plt.suptitle('MaHealthBiasAudit v2 - Comprehensive Bias Dashboard\n'
                    'English | Swahili | Yoruba | Amharic', 
                    fontsize=16, fontweight='bold')
        
        self._save_and_show(fig, "comprehensive_dashboard.png", dpi=150)
        
        return fig
    
    def get_generated_figures(self) -> List[Dict]:
        """Return list of generated figures"""
        return self.generated_figures


# Test the dashboard
if __name__ == "__main__":
    dashboard = BiasVisualizationDashboard(save_figures=True, show_display=False)
    
    # Create dummy data
    languages = ['English', 'Swahili', 'Yoruba', 'Amharic']
    sdi_matrix = pd.DataFrame(np.random.rand(4, 4) * 0.5, index=languages, columns=languages)
    np.fill_diagonal(sdi_matrix.values, 0)
    
    dashboard.plot_sdi_heatmap(sdi_matrix)
    
    print("\n✅ Visualization dashboard test complete!")