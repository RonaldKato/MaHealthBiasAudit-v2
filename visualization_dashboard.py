"""
Visualization Dashboard for MaHealthBiasAudit v2
Visualizations for bias audit reports
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# For inline display in notebooks/scripts
try:
    from IPython.display import display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False


class BiasVisualizationDashboard:
    """
    Handles all visualization creation for the bias audit pipeline
    Automatically displays and saves figures
    """
    
    def __init__(self, save_figures: bool = True, output_dir: str = "output", show_display: bool = True):
        """
        Initialize the visualization dashboard
        
        Args:
            save_figures: Whether to save figures to disk
            output_dir: Directory to save figures
            show_display: Whether to show figures interactively
        """
        self.save_figures = save_figures
        self.output_dir = output_dir
        self.show_display = show_display
        
        # Create output directory if it doesn't exist
        if self.save_figures:
            os.makedirs(output_dir, exist_ok=True)
        
        # Store generated figures for later display
        self.generated_figures = []
        
        print(f"Visualization Dashboard initialized")
        print(f"   Save figures: {save_figures} → {output_dir}")
        print(f"   Show display: {show_display}")
    
    def _save_and_show(self, fig, filename: str, dpi: int = 150, close_after: bool = False):
        """Helper to save and/or show figure"""
        if self.save_figures:
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"  Saved: {filepath}")
        
        if self.show_display:
            plt.figure(fig.number)
            plt.show()
            
            # Also try IPython display if available
            if IPYTHON_AVAILABLE:
                display(fig)
        
        self.generated_figures.append({'fig': fig, 'filename': filename})
        
        if close_after:
            plt.close(fig)
        
        return fig
    
    def plot_3d_embedding_space(self, embeddings: np.ndarray, 
                                 language_labels: List[str], 
                                 topic_labels: List[str] = None,
                                 title: str = "Multilingual Embedding Space") -> plt.Figure:
        """
        Create 3D t-SNE visualization of embedding space
        """
        print("\n  Creating 3D embedding space visualization...")
        
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
        self._save_and_show(fig, "3d_embedding_space.png", dpi=150)
        
        return fig
    
    def plot_sdi_heatmap(self, sdi_matrix: pd.DataFrame, 
                         title: str = "Semantic Divergence Index (SDI)") -> plt.Figure:
        """
        Create heatmap of Semantic Divergence Index between language pairs
        """
        print("\n  Creating SDI heatmap...")
        
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
        avg_sdi = sdi_matrix.values[np.triu_indices_from(sdi_matrix.values, k=1)].mean()
        fig.text(0.02, 0.02, f"Average SDI: {avg_sdi:.3f} | Thresholds: >0.2=moderate, >0.4=high", 
                fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        self._save_and_show(fig, "sdi_heatmap.png")
        
        return fig
    
    def plot_tokenisation_parity(self, tp_df: pd.DataFrame) -> plt.Figure:
        """
        Create visualization of tokenisation parity across languages
        """
        print("\n  Creating tokenisation parity visualization...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Fertility Penalty by Tokeniser
        if 'Fertility_Penalty_mBERT' in tp_df.columns:
            # Prepare data for grouped bar chart
            languages = tp_df['language'].tolist()
            x = np.arange(len(languages))
            width = 0.25
            
            axes[0].bar(x - width, tp_df['Fertility_Penalty_mBERT'], width, label='mBERT', color='skyblue')
            axes[0].bar(x, tp_df['Fertility_Penalty_XLM-R'], width, label='XLM-R', color='lightgreen')
            axes[0].bar(x + width, tp_df['Fertility_Penalty_AfriBERTa'], width, label='AfriBERTa', color='coral')
            
            axes[0].axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Threshold (1.5)')
            axes[0].set_title('Fertility Penalty by Tokeniser\n(higher = more tokens needed)', fontweight='bold')
            axes[0].set_ylabel('Fertility Penalty')
            axes[0].set_xlabel('Language')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(languages, rotation=45, ha='right')
            axes[0].legend()
            axes[0].set_ylim(0, 3)
        
        # Plot 2: OOV Rate
        if 'OOV_Rate' in tp_df.columns:
            bars = axes[1].bar(tp_df['language'], tp_df['OOV_Rate'] * 100, 
                              color='lightcoral', edgecolor='black')
            axes[1].axhline(y=15, color='red', linestyle='--', linewidth=2, label='Threshold (15%)')
            axes[1].set_title('Out-of-Vocabulary Rate\n(unknown tokens)', fontweight='bold')
            axes[1].set_ylabel('OOV Rate (%)')
            axes[1].set_xlabel('Language')
            axes[1].legend()
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, tp_df['OOV_Rate'] * 100):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Best Tokeniser
        if 'Best_Tokeniser' in tp_df.columns:
            best_counts = tp_df['Best_Tokeniser'].value_counts()
            colors = {'mBERT': 'skyblue', 'XLM-R': 'lightgreen', 'AfriBERTa': 'coral'}
            bar_colors = [colors.get(t, 'gray') for t in best_counts.index]
            
            bars = axes[2].bar(best_counts.index, best_counts.values, color=bar_colors, edgecolor='black')
            axes[2].set_title('Best Tokeniser per Language\n(lowest fertility penalty)', fontweight='bold')
            axes[2].set_ylabel('Number of Languages')
            axes[2].set_xlabel('Tokeniser')
            
            # Add value labels
            for bar, value in zip(bars, best_counts.values):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           str(value), ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Tokenisation Parity Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save_and_show(fig, "tokenisation_parity.png")
        
        return fig
    
    def plot_bias_pattern_heatmap(self, patterns_df: pd.DataFrame) -> plt.Figure:
        """
        Create heatmap of bias patterns by topic
        """
        print("\n  Creating bias patterns heatmap...")
        
        if patterns_df.empty or 'topic' not in patterns_df.columns:
            print("     No bias pattern data to plot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data for heatmap
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
        ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Moderate (0.2)')
        ax.axvline(x=0.4, color='red', linestyle='--', linewidth=2, alpha=0.7, label='High (0.4)')
        
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
        
        plt.tight_layout()
        self._save_and_show(fig, "bias_patterns.png")
        
        return fig
    
    def plot_interrogative_analysis(self, inter_df: pd.DataFrame) -> plt.Figure:
        """
        Create visualization of interrogative structure analysis
        """
        print("\n  Creating interrogative structure analysis...")
        
        if inter_df.empty:
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
        print("\n  Creating Trust-Aware Module visualization...")
        
        if not trust_results:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Trust scores by language
        languages = list(trust_results.keys())
        trust_scores = [trust_results[l].trust_score for l in languages]
        preservation_needed = [trust_results[l].preservation_needed for l in languages]
        
        colors = ['coral' if p else 'skyblue' for p in preservation_needed]
        bars = axes[0].bar(languages, trust_scores, color=colors, edgecolor='black')
        axes[0].axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='High Trust')
        axes[0].axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Medium Trust')
        axes[0].set_title('Trust-Aware Analysis\n(higher = more culturally appropriate)', fontweight='bold')
        axes[0].set_ylabel('Trust Score')
        axes[0].set_xlabel('Language')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
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
        
        # Add value labels
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
        print("\n  Creating RCA summary visualization...")
        
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
        
        # Add value labels
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Preservation vs Intervention
        intervention = len(rca_results) - preserve_counts
        labels = ['Preserve (Cultural)', 'Intervention Needed']
        sizes = [preserve_counts, intervention]
        colors = ['green', 'red']
        
        axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
                   explode=(0.05, 0), shadow=True)
        axes[1].set_title('Bias Resolution Strategy\n(Cultural vs Technical)', fontweight='bold')
        
        plt.suptitle('Root Cause Attribution (RCA) Cascade Results', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        self._save_and_show(fig, "rca_summary.png")
        
        return fig
    
    def create_dashboard(self, sdi_matrix: pd.DataFrame, 
                        tp_df: pd.DataFrame,
                        performance_df: pd.DataFrame,
                        morph_results: Dict,
                        inter_df: pd.DataFrame = None,
                        trust_results: Dict = None,
                        rca_results: List = None,
                        embed_figure: plt.Figure = None) -> plt.Figure:
        """
        Create a comprehensive dashboard combining multiple visualizations
        """
        print("\n  Creating comprehensive dashboard...")
        
        # Create a large figure for the dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Determine grid layout based on available data
        has_extra = inter_df is not None or trust_results or rca_results
        
        if embed_figure:
            gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        else:
            gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        plot_idx = 1
        
        # 1. SDI Heatmap (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if not sdi_matrix.empty:
            im = ax1.imshow(sdi_matrix.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
            ax1.set_xticks(range(len(sdi_matrix.columns)))
            ax1.set_yticks(range(len(sdi_matrix.index)))
            ax1.set_xticklabels(sdi_matrix.columns, rotation=45, ha='right', fontsize=8)
            ax1.set_yticklabels(sdi_matrix.index, fontsize=8)
            ax1.set_title('SDI Heatmap\n(higher = more bias)', fontweight='bold', fontsize=10)
            plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # 2. Fertility Penalty (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        if not tp_df.empty and 'Overall_Fertility_Penalty' in tp_df.columns:
            languages = tp_df['language'].tolist()
            penalties = tp_df['Overall_Fertility_Penalty'].tolist()
            colors = ['red' if p > 1.5 else 'orange' if p > 1.2 else 'green' for p in penalties]
            bars = ax2.bar(languages, penalties, color=colors, edgecolor='black')
            ax2.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='High Threshold')
            ax2.set_title('Tokenisation Fertility Penalty\n(>1.5 = significant bias)', fontweight='bold', fontsize=10)
            ax2.set_ylabel('Fertility Penalty')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(fontsize=8)
            ax2.set_ylim(0, 3)
            
            for bar, val in zip(bars, penalties):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 3. OOV Rate (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if not tp_df.empty and 'OOV_Rate' in tp_df.columns:
            languages = tp_df['language'].tolist()
            oov_rates = (tp_df['OOV_Rate'] * 100).tolist()
            colors = ['red' if o > 15 else 'orange' if o > 10 else 'green' for o in oov_rates]
            bars = ax3.bar(languages, oov_rates, color=colors, edgecolor='black')
            ax3.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Threshold (15%)')
            ax3.set_title('Out-of-Vocabulary Rate\n(unknown tokens)', fontweight='bold', fontsize=10)
            ax3.set_ylabel('OOV Rate (%)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend(fontsize=8)
            
            for bar, val in zip(bars, oov_rates):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 4. Interrogative Analysis (if available)
        if inter_df is not None and not inter_df.empty:
            ax4 = fig.add_subplot(gs[1, 0])
            mismatch_by_lang = inter_df.groupby('Language')['Mismatch'].sum()
            if not mismatch_by_lang.empty:
                colors = ['red' if v > 0 else 'green' for v in mismatch_by_lang.values]
                bars = ax4.bar(mismatch_by_lang.index, mismatch_by_lang.values, color=colors, edgecolor='black')
                ax4.set_title('Query Structure Mismatch\n(Non-English patterns)', fontweight='bold', fontsize=10)
                ax4.set_ylabel('Mismatches')
                ax4.tick_params(axis='x', rotation=45)
                for bar, val in zip(bars, mismatch_by_lang.values):
                    if val > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                str(int(val)), ha='center', va='bottom', fontsize=8)
        
        # 5. Trust Scores (if available)
        if trust_results:
            ax5 = fig.add_subplot(gs[1, 1])
            languages = list(trust_results.keys())
            trust_scores = [trust_results[l].trust_score for l in languages]
            colors = ['green' if t > 0.7 else 'orange' if t > 0.5 else 'red' for t in trust_scores]
            bars = ax5.bar(languages, trust_scores, color=colors, edgecolor='black')
            ax5.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='High Trust')
            ax5.set_title('Trust-Aware Score\n(cultural appropriateness)', fontweight='bold', fontsize=10)
            ax5.set_ylabel('Trust Score')
            ax5.set_ylim(0, 1)
            ax5.tick_params(axis='x', rotation=45)
            ax5.legend(fontsize=8)
            for bar, val in zip(bars, trust_scores):
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 6. RCA Summary (if available)
        if rca_results:
            ax6 = fig.add_subplot(gs[1, 2])
            cause_counts = {}
            for result in rca_results:
                cause_counts[result.root_cause] = cause_counts.get(result.root_cause, 0) + 1
            causes = list(cause_counts.keys())
            counts = list(cause_counts.values())
            colors = {'TOKENISATION': 'skyblue', 'MORPHOLOGY': 'lightgreen', 
                      'QUERY_STRUCTURE': 'orange', 'CULTURAL': 'coral', 'UNKNOWN': 'gray'}
            bar_colors = [colors.get(c, 'gray') for c in causes]
            bars = ax6.bar(causes, counts, color=bar_colors, edgecolor='black')
            ax6.set_title('Root Cause Attribution\n(bias sources)', fontweight='bold', fontsize=10)
            ax6.tick_params(axis='x', rotation=45)
            for bar, val in zip(bars, counts):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(val), ha='center', va='bottom', fontsize=8)
        
        # 7. Bias Patterns (if available)
        if 'bias_patterns' in self.__dict__ and self.bias_patterns is not None:
            ax7 = fig.add_subplot(gs[2, 0] if has_extra else gs[2, :2])
            patterns_df = self.bias_patterns
            topics = patterns_df['topic'].tolist()
            sdi_values = patterns_df['avg_sdi'].tolist()
            severities = patterns_df['bias_severity'].tolist()
            color_map = {'low': 'green', 'moderate': 'orange', 'high': 'red'}
            bar_colors = [color_map.get(s, 'blue') for s in severities]
            
            y_pos = np.arange(len(topics))
            bars = ax7.barh(y_pos, sdi_values, color=bar_colors, edgecolor='black')
            ax7.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, label='Moderate')
            ax7.axvline(x=0.4, color='red', linestyle='--', alpha=0.7, label='High')
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels([t.replace('_', ' ').title() for t in topics])
            ax7.set_xlabel('SDI')
            ax7.set_title('Bias by Maternal Health Topic', fontweight='bold', fontsize=10)
            ax7.legend(fontsize=8)
        
        # 8. 3D Embedding (if available)
        if embed_figure and embed_figure.axes:
            ax_embed = fig.add_subplot(gs[2 if has_extra else 1, 2], projection='3d')
            # Copy the content from embed_figure
            for child in embed_figure.axes[0].collections:
                ax_embed.add_collection(child)
            ax_embed.set_xlabel('t-SNE 1', fontsize=8)
            ax_embed.set_ylabel('t-SNE 2', fontsize=8)
            ax_embed.set_zlabel('t-SNE 3', fontsize=8)
            ax_embed.set_title('Embedding Space\n(clusters = bias)', fontweight='bold', fontsize=10)
        
        plt.suptitle('MaHealthBiasAudit v2 - Complete Bias Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        self._save_and_show(fig, "complete_dashboard.png", dpi=150)
        
        return fig
    
    def show_all_figures(self):
        """Display all generated figures"""
        print(f"\n  Generated {len(self.generated_figures)} figures:")
        for i, fig_info in enumerate(self.generated_figures):
            print(f"     {i+1}. {fig_info['filename']}")
        
        if self.show_display and self.generated_figures:
            print("\n  Displaying all figures...")
            for fig_info in self.generated_figures:
                plt.figure(fig_info['fig'].number)
                plt.show()
    
    def close_all(self):
        """Close all figure windows"""
        plt.close('all')
        print("  Closed all figure windows")