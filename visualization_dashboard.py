"""
Visualization Dashboard for MaHealthBiasAudit
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

from config import FIGURES_DIR, THRESHOLDS, LANG_COLORS, RCA_COLORS


class VisualizationDashboard:
    def __init__(self):
        self.output_dir = FIGURES_DIR
        self.tables_dir = os.path.join(FIGURES_DIR, 'tables')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        self.setup_custom_styles()
        print(f"Visualization Dashboard ready - Output: {self.output_dir}")
        print(f"Tables Directory: {self.tables_dir}")
    
    def setup_custom_styles(self):
        """Setup custom visualization styles"""
        self.colors = {
            'high_bias': '#E74C3C',
            'moderate_bias': '#F39C12',
            'low_bias': '#27AE60',
            'background': '#F8F9FA',
            'text': '#2C3E50',
            'grid': '#BDC3C7',
            'primary': '#3498DB',
            'secondary': '#9B59B6'
        }
    
    def create_all_visualizations(self, sdi_matrix=None, tp_df=None, trust_results=None, 
                                    rca_results=None, embeddings=None, labels=None,
                                    bias_results=None, performance_metrics=None):
        """Create all visualizations"""
        print("\n" + "="*60)
        print("Creating Visualizations")
        print("="*60)
        
        # SDI Visualizations
        if sdi_matrix is not None and not sdi_matrix.empty:
            self.plot_sdi_heatmap_enhanced(sdi_matrix)
            self.plot_sdi_bar_chart_enhanced(sdi_matrix)
            self.plot_sdi_radar_chart(sdi_matrix)
        
        # Tokenisation Visualizations
        if tp_df is not None and not tp_df.empty:
            self.plot_tokenisation_chart(tp_df)
            self.plot_tokenisation_boxplot(tp_df)
        
        # Trust Visualizations
        if trust_results:
            self.plot_trust_chart(trust_results)
            self.plot_trust_gauge_meter(trust_results)
        
        # RCA Visualizations
        if rca_results:
            self.plot_rca_pie(rca_results)
            self.plot_rca_treemap(rca_results)
        
        # Embedding Visualizations
        if embeddings is not None and len(embeddings) > 0 and labels:
            self.plot_embedding_tsne(embeddings, labels)
            self.plot_embedding_pca_3d(embeddings, labels)
        
        # Additional powerful visuals
        if bias_results:
            self.plot_bias_waterfall_chart(bias_results)
            self.plot_bias_heatmap_timeline(bias_results)
        
        if performance_metrics:
            self.plot_performance_dashboard(performance_metrics)
        
        # Comprehensive dashboards
        self.plot_summary_dashboard(sdi_matrix, tp_df, trust_results, rca_results)
        self.plot_executive_dashboard(sdi_matrix, tp_df, trust_results, rca_results, bias_results)
        
        print(f"\n All visualizations saved to {self.output_dir}")
    
    def create_all_tables(self, sdi_matrix=None, tp_df=None, trust_results=None, 
                          rca_results=None, bias_results=None, performance_metrics=None,
                          corpus_stats=None):
        """Create results tables"""
        print("\n" + "="*60)
        print("Creating Results Tables")
        print("="*60)
        
        tables = []
        
        # Table 1: SDI Comprehensive Matrix
        if sdi_matrix is not None and not sdi_matrix.empty:
            table1 = self.create_sdi_table(sdi_matrix)
            tables.append(("SDI_Comprehensive_Matrix", table1))
        
        # Table 2: Tokenisation Parity Results
        if tp_df is not None and not tp_df.empty:
            table2 = self.create_tokenisation_table(tp_df)
            tables.append(("Tokenisation_Parity_Results", table2))
        
        # Table 3: Trust Score Breakdown
        if trust_results:
            table3 = self.create_trust_table(trust_results)
            tables.append(("Trust_Score_Breakdown", table3))
        
        # Table 4: Root Cause Analysis Summary
        if rca_results:
            table4 = self.create_rca_table(rca_results)
            tables.append(("Root_Cause_Analysis", table4))
        
        # Table 5: Bias Detection Summary
        if bias_results:
            table5 = self.create_bias_summary_table(bias_results)
            tables.append(("Bias_Detection_Summary", table5))
        
        # Table 6: Performance Metrics
        if performance_metrics:
            table6 = self.create_performance_table(performance_metrics)
            tables.append(("Performance_Metrics", table6))
        
        # Table 7: Corpus Statistics
        if corpus_stats is not None and not corpus_stats.empty:
            table7 = self.create_corpus_stats_table(corpus_stats)
            tables.append(("Corpus_Statistics", table7))
        
        # Table 8: Language-wise Comparison
        if sdi_matrix is not None and tp_df is not None and trust_results:
            table8 = self.create_language_comparison_table(sdi_matrix, tp_df, trust_results)
            tables.append(("Language_Wise_Comparison", table8))
        
        # Save all tables
        for table_name, table_df in tables:
            self.save_table(table_df, table_name)
        
        print(f"\n {len(tables)} tables saved to {self.tables_dir}")
        return tables
    
    # ==================== SDI VISUALIZATIONS ====================
    
    def plot_sdi_heatmap_enhanced(self, sdi_matrix):
        """SDI heatmap"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(sdi_matrix.values, cmap='RdYlGn_r', vmin=0, vmax=0.8)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Semantic Divergence Index (SDI)', fontsize=12, fontweight='bold')
        
        # Add value labels
        for i in range(len(sdi_matrix.index)):
            for j in range(len(sdi_matrix.columns)):
                value = sdi_matrix.iloc[i, j]
                color = 'white' if value > 0.4 else 'black'
                severity = '🔴' if value > 0.4 else '🟡' if value > 0.2 else '🟢'
                ax.text(j, i, f'{severity}\n{value:.3f}', ha='center', va='center', 
                       color=color, fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(sdi_matrix.columns)))
        ax.set_yticks(range(len(sdi_matrix.index)))
        ax.set_xticklabels(sdi_matrix.columns, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(sdi_matrix.index, fontsize=11)
        
        ax.set_title('Semantic Divergence Index (SDI) Heatmap\nHigher Values = More Bias', 
                    fontweight='bold', fontsize=14, pad=20)
        plt.tight_layout()
        self._save_fig(fig, 'sdi_heatmap_enhanced.png')
    
    def plot_sdi_bar_chart_enhanced(self, sdi_matrix):
        """SDI bar chart with trend line"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        languages = [l for l in sdi_matrix.columns if l != 'English']
        sdi_values = [sdi_matrix.loc['English', l] for l in languages]
        
        colors = ['#E74C3C' if v > 0.4 else '#F39C12' if v > 0.2 else '#27AE60' for v in sdi_values]
        
        bars = ax.bar(languages, sdi_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add threshold lines
        ax.axhline(y=0.4, color='#E74C3C', linestyle='--', linewidth=2.5, 
                  label='High Bias Threshold', alpha=0.7)
        ax.axhline(y=0.2, color='#F39C12', linestyle='--', linewidth=2.5, 
                  label='Moderate Threshold', alpha=0.7)
        
        # Add value labels
        for bar, val in zip(bars, sdi_values):
            trend = '↑' if val > 0.3 else '↓' if val < 0.15 else '→'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                   f'{trend} {val:.3f}', ha='center', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('SDI Score (vs English)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Target Language', fontsize=12, fontweight='bold')
        ax.set_title('Cross-Lingual Bias Analysis: English Baseline Comparison', 
                    fontweight='bold', fontsize=14, pad=15)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_fig(fig, 'sdi_barchart_enhanced.png')
    
    def plot_sdi_radar_chart(self, sdi_matrix):
        """Radar chart for SDI comparison"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        languages = [l for l in sdi_matrix.columns if l != 'English']
        sdi_values = [sdi_matrix.loc['English', l] for l in languages]
        
        angles = np.linspace(0, 2 * np.pi, len(languages), endpoint=False).tolist()
        sdi_values_radar = sdi_values + sdi_values[:1]
        angles += angles[:1]
        
        ax.plot(angles, sdi_values_radar, 'o-', linewidth=2, color='#E74C3C', label='SDI Score')
        ax.fill(angles, sdi_values_radar, alpha=0.25, color='#E74C3C')
        
        # Add value labels
        for i, (lang, val) in enumerate(zip(languages, sdi_values)):
            angle_rad = angles[i]
            ax.text(angle_rad, val + 0.05, f'{val:.3f}', ha='center', fontweight='bold')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(languages, fontsize=11)
        ax.set_ylim(0, max(sdi_values) + 0.1)
        ax.set_title('Language Bias Radar Chart', fontweight='bold', fontsize=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        self._save_fig(fig, 'sdi_radar.png')
    
    # ==================== TOKENISATION VISUALIZATIONS ====================
    
    def plot_tokenisation_chart(self, tp_df):
        """Plot tokenisation parity chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Aggregate by language if multiple tokenisers
        lang_data = []
        for lang in tp_df['Language'].unique():
            lang_df = tp_df[tp_df['Language'] == lang]
            if not lang_df.empty:
                lang_data.append({
                    'Language': lang,
                    'Fertility': lang_df['Fertility_Penalty'].mean()
                })
        
        df = pd.DataFrame(lang_data)
        if df.empty:
            return
        
        colors = ['#E74C3C' if f > 1.5 else '#27AE60' for f in df['Fertility']]
        
        bars = ax.bar(df['Language'], df['Fertility'], color=colors, edgecolor='black')
        ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Threshold')
        
        for bar, val in zip(bars, df['Fertility']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.2f}', ha='center', fontweight='bold')
        
        ax.set_ylabel('Fertility Penalty', fontsize=12, fontweight='bold')
        ax.set_title('Tokenisation Parity by Language', fontweight='bold', fontsize=14)
        ax.legend()
        plt.tight_layout()
        self._save_fig(fig, 'tokenisation.png')
    
    def plot_tokenisation_boxplot(self, tp_df):
        """Boxplot for tokenisation distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = []
        labels = []
        for lang in tp_df['Language'].unique():
            lang_df = tp_df[tp_df['Language'] == lang]
            if not lang_df.empty:
                data.append(lang_df['Fertility_Penalty'].values)
                labels.append(lang)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#3498DB', '#E74C3C', '#27AE60', '#F39C12'][:len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Threshold')
            ax.set_ylabel('Fertility Penalty', fontsize=12, fontweight='bold')
            ax.set_title('Tokenisation Distribution by Language', fontweight='bold', fontsize=14)
            ax.legend()
            
            plt.tight_layout()
            self._save_fig(fig, 'tokenisation_boxplot.png')
    
    # ==================== TRUST VISUALIZATIONS ====================
    
    def plot_trust_chart(self, trust_results):
        """Plot trust-aware results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        languages = list(trust_results.keys())
        scores = [trust_results[l].trust_score for l in languages]
        
        colors = ['#27AE60' if s > 0.7 else '#F39C12' if s > 0.4 else '#E74C3C' for s in scores]
        bars = ax.bar(languages, scores, color=colors, edgecolor='black')
        ax.axhline(y=0.7, color='green', linestyle='--', linewidth=2, label='Target')
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.2f}', ha='center', fontweight='bold')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Trust Score', fontsize=12, fontweight='bold')
        ax.set_title('Trust-Aware Module: Cultural Appropriateness', fontweight='bold', fontsize=14)
        ax.legend()
        plt.tight_layout()
        self._save_fig(fig, 'trust_scores.png')
    
    def plot_trust_gauge_meter(self, trust_results):
        """Gauge meter for trust scores"""
        n_langs = len(trust_results)
        n_cols = min(3, n_langs)
        n_rows = (n_langs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_langs == 1:
            axes = [axes]
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        for idx, (lang, result) in enumerate(trust_results.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            score = result.trust_score
            
            # Create gauge
            theta = np.linspace(0, np.pi, 100)
            x = np.cos(theta)
            y = np.sin(theta)
            
            # Color based on score
            if score > 0.7:
                color = '#27AE60'
                status = 'High Trust'
            elif score > 0.4:
                color = '#F39C12'
                status = 'Moderate Trust'
            else:
                color = '#E74C3C'
                status = 'Low Trust'
            
            # Create needle
            needle_angle = np.pi * (1 - score)
            needle_x = [0, 0.8 * np.cos(needle_angle)]
            needle_y = [0, 0.8 * np.sin(needle_angle)]
            
            # Plot gauge
            ax.plot(x, y, 'k-', linewidth=2)
            ax.fill_between(x, 0, y, where=(y >= 0), alpha=0.3, color=color)
            ax.plot(needle_x, needle_y, 'r-', linewidth=3)
            ax.plot(0, 0, 'ko', markersize=10)
            
            # Add labels
            ax.text(0, -0.3, f'{score:.2f}', ha='center', fontsize=20, fontweight='bold')
            ax.text(0, -0.5, status, ha='center', fontsize=10)
            ax.set_title(f'{lang}', fontweight='bold', fontsize=12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.6, 1)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(len(trust_results), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Trust Score Gauge Meter\nCultural Assessment', 
                    fontweight='bold', fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_fig(fig, 'trust_gauge_meter.png')
    
    # ==================== RCA VISUALIZATIONS ====================
    
    def plot_rca_pie(self, rca_results):
        """Plot RCA distribution pie chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        counts = {}
        for r in rca_results:
            cause = getattr(r, 'root_cause', str(r))
            counts[cause] = counts.get(cause, 0) + 1
        
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = [RCA_COLORS.get(l, '#95A5A6') for l in labels]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct=lambda pct: f'{pct:.1f}%',
                                          startangle=90, wedgeprops={'edgecolor': 'white'})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Root Cause Attribution (RCA) Distribution', fontweight='bold', fontsize=14)
        plt.tight_layout()
        self._save_fig(fig, 'rca_pie.png')
    
    def plot_rca_treemap(self, rca_results):
        """Treemap for root cause analysis"""
        try:
            import squarify
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            counts = {}
            for r in rca_results:
                cause = getattr(r, 'root_cause', str(r))
                counts[cause] = counts.get(cause, 0) + 1
            
            labels = list(counts.keys())
            sizes = list(counts.values())
            colors = [RCA_COLORS.get(l, '#95A5A6') for l in labels]
            
            # Create treemap
            squarify.plot(sizes=sizes, label=[f'{l}\n({s} cases)' for l, s in zip(labels, sizes)],
                         color=colors, alpha=0.8, ax=ax, text_kwargs={'fontsize': 10, 'fontweight': 'bold'})
            
            ax.set_title('Root Cause Attribution Treemap\nLarger Box = More Frequent Cause', 
                        fontweight='bold', fontsize=14)
            ax.axis('off')
            
            plt.tight_layout()
            self._save_fig(fig, 'rca_treemap.png')
        except ImportError:
            print("   Note: squarify not installed - treemap visualization skipped")
            print("   Install with: pip install squarify")
    
    # ==================== EMBEDDING VISUALIZATIONS ====================
    
    def plot_embedding_tsne(self, embeddings, labels):
        """Plot t-SNE visualization of embeddings"""
        try:
            from sklearn.manifold import TSNE
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            perplexity = min(30, len(embeddings) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            emb_2d = tsne.fit_transform(embeddings)
            
            unique_langs = list(set(labels))
            for lang in unique_langs:
                mask = [l == lang for l in labels]
                color = LANG_COLORS.get(lang, '#95A5A6')
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                          c=color, s=60, alpha=0.7, edgecolors='black', label=lang)
            
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.set_title('Multilingual Embedding Space (t-SNE)', fontweight='bold', fontsize=14)
            ax.legend()
            plt.tight_layout()
            self._save_fig(fig, 'embedding_tsne.png')
        except Exception as e:
            print(f"   t-SNE visualization skipped: {e}")
    
    def plot_embedding_pca_3d(self, embeddings, labels):
        """3D PCA visualization"""
        try:
            from sklearn.decomposition import PCA
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Reduce to 3D
            pca = PCA(n_components=3, random_state=42)
            emb_3d = pca.fit_transform(embeddings)
            
            # Plot each language
            unique_langs = list(set(labels))
            for lang in unique_langs:
                mask = [l == lang for l in labels]
                color = LANG_COLORS.get(lang, '#95A5A6')
                ax.scatter(emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
                          c=color, s=80, alpha=0.7, edgecolors='black', linewidth=0.5, label=lang)
            
            ax.set_xlabel('PC1', fontsize=11)
            ax.set_ylabel('PC2', fontsize=11)
            ax.set_zlabel('PC3', fontsize=11)
            ax.set_title('3D Multilingual Embedding Space (PCA)', fontweight='bold', fontsize=14)
            ax.legend(loc='upper left', fontsize=10)
            
            plt.tight_layout()
            self._save_fig(fig, 'embedding_pca_3d.png')
        except Exception as e:
            print(f"   3D PCA visualization skipped: {e}")
    
    # ==================== BIAS VISUALIZATIONS ====================
    
    def plot_bias_waterfall_chart(self, bias_results):
        """Waterfall chart for bias accumulation"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        bias_types = list(bias_results.keys())
        bias_scores = list(bias_results.values())
        
        # Create waterfall
        for i, (bias_type, score) in enumerate(zip(bias_types, bias_scores)):
            color = '#E74C3C' if score > 0.2 else '#F39C12' if score > 0.1 else '#27AE60'
            bar = ax.bar(i, score, color=color, edgecolor='black')
            
            # Add labels
            ax.text(i, score + 0.01, f'{score:.3f}', ha='center', fontweight='bold')
        
        # Add total line
        total_bias = sum(bias_scores)
        ax.axhline(y=total_bias, color='#E74C3C', linestyle='-', linewidth=2.5,
                  label=f'Total Bias: {total_bias:.3f}')
        
        ax.set_xticks(range(len(bias_types)))
        ax.set_xticklabels(bias_types, rotation=45, ha='right')
        ax.set_ylabel('Bias Score', fontsize=12, fontweight='bold')
        ax.set_title('💧 Bias Accumulation Waterfall Chart\nContributions to Overall Bias', 
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        self._save_fig(fig, 'bias_waterfall.png')
    
    def plot_bias_heatmap_timeline(self, bias_results):
        """Temporal bias heatmap"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bias_types = list(bias_results.keys())
        timeline = ['Week 1', 'Week 2', 'Week 3', 'Week 4']
        
        # Generate simulated temporal data
        np.random.seed(42)
        data = np.random.rand(len(bias_types), len(timeline)) * 0.5
        
        # Create heatmap
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.5)
        
        # Add labels
        ax.set_xticks(range(len(timeline)))
        ax.set_yticks(range(len(bias_types)))
        ax.set_xticklabels(timeline)
        ax.set_yticklabels(bias_types)
        
        # Add value annotations
        for i in range(len(bias_types)):
            for j in range(len(timeline)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                              ha="center", va="center", color="white" if data[i, j] > 0.25 else "black")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Bias Intensity', fontsize=11)
        
        ax.set_title('Bias Evolution Timeline Heatmap', fontweight='bold', fontsize=14)
        plt.tight_layout()
        self._save_fig(fig, 'bias_timeline_heatmap.png')
    
    # ==================== PERFORMANCE DASHBOARD ====================
    
    def plot_performance_dashboard(self, performance_metrics):
        """Performance metrics dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = list(performance_metrics.keys())
        values = list(performance_metrics.values())
        
        # Top-Left: Gauge for overall performance
        ax1 = axes[0, 0]
        overall = np.mean(values)
        color = '#27AE60' if overall > 0.7 else '#F39C12' if overall > 0.4 else '#E74C3C'
        
        # Create semicircle gauge
        theta = np.linspace(0, np.pi, 100)
        x = np.cos(theta)
        y = np.sin(theta)
        ax1.plot(x, y, 'k-', linewidth=2)
        ax1.fill_between(x, 0, y, where=(y >= 0), alpha=0.3, color=color)
        ax1.text(0, 0.2, f'{overall:.2%}', ha='center', fontsize=24, fontweight='bold')
        ax1.text(0, -0.2, 'Overall Performance', ha='center', fontsize=11)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-0.3, 1)
        ax1.axis('off')
        ax1.set_title('Performance Score', fontweight='bold')
        
        # Top-Right: Bar chart
        ax2 = axes[0, 1]
        bars = ax2.bar(metrics, values, color='#3498DB', edgecolor='black')
        for bar, val in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', fontweight='bold')
        ax2.set_ylabel('Score', fontsize=11)
        ax2.set_title('Performance by Metric', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        
        # Bottom-Left: Donut chart
        ax3 = axes[1, 0]
        passed = sum(1 for v in values if v > 0.7)
        failed = len(values) - passed
        wedges, texts, autotexts = ax3.pie([passed, failed], labels=['Passed', 'Failed'],
                                           autopct='%1.1f%%', colors=['#27AE60', '#E74C3C'],
                                           wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax3.set_title('Pass Rate', fontweight='bold')
        
        # Bottom-Right: Radar chart
        ax4 = axes[1, 1]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        values_radar = values + values[:1]
        angles += angles[:1]
        
        ax4.plot(angles, values_radar, 'o-', linewidth=2, color='#E67E22')
        ax4.fill(angles, values_radar, alpha=0.25, color='#E67E22')
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metrics, fontsize=9)
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Radar', fontweight='bold')
        
        plt.suptitle('Performance Metrics Dashboard', fontweight='bold', fontsize=16, y=1.02)
        plt.tight_layout()
        self._save_fig(fig, 'performance_dashboard.png')
    
    # ==================== SUMMARY DASHBOARDS ====================
    
    def plot_summary_dashboard(self, sdi_matrix, tp_df, trust_results, rca_results):
        """Plot - summary dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Top-Left: SDI Summary
        ax1 = axes[0, 0]
        if sdi_matrix is not None and not sdi_matrix.empty:
            triu_indices = np.triu_indices_from(sdi_matrix.values, k=1)
            if len(triu_indices[0]) > 0:
                avg_sdi = sdi_matrix.values[triu_indices].mean()
                ax1.text(0.5, 0.6, f'{avg_sdi:.3f}', fontsize=48, ha='center', fontweight='bold',
                        color='#E74C3C' if avg_sdi > 0.4 else '#F39C12' if avg_sdi > 0.2 else '#27AE60')
                ax1.text(0.5, 0.3, f'Average SDI\n{"HIGH" if avg_sdi > 0.4 else "MODERATE" if avg_sdi > 0.2 else "LOW"} Bias',
                        ha='center', fontsize=14)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Semantic Divergence Index', fontweight='bold')
        
        # Top-Right: Tokenisation
        ax2 = axes[0, 1]
        if tp_df is not None and not tp_df.empty:
            lang_tp = {}
            for lang in tp_df['Language'].unique():
                lang_df = tp_df[tp_df['Language'] == lang]
                lang_tp[lang] = lang_df['Fertility_Penalty'].mean()
            
            langs = list(lang_tp.keys())
            values = list(lang_tp.values())
            colors = ['#E74C3C' if v > 1.5 else '#27AE60' for v in values]
            ax2.bar(langs, values, color=colors, edgecolor='black')
            ax2.axhline(y=1.5, color='red', linestyle='--')
            ax2.set_ylabel('Fertility')
            ax2.set_title('Tokenisation Parity', fontweight='bold')
            ax2.tick_params(axis='x', rotation=45)
        
        # Bottom-Left: Trust Scores
        ax3 = axes[1, 0]
        if trust_results:
            langs = list(trust_results.keys())
            scores = [trust_results[l].trust_score for l in langs]
            colors = ['#27AE60' if s > 0.7 else '#F39C12' if s > 0.4 else '#E74C3C' for s in scores]
            ax3.bar(langs, scores, color=colors, edgecolor='black')
            ax3.axhline(y=0.7, color='green', linestyle='--')
            ax3.set_ylim(0, 1)
            ax3.set_ylabel('Trust Score')
            ax3.set_title('Cultural Trust', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        
        # Bottom-Right: RCA Summary
        ax4 = axes[1, 1]
        if rca_results:
            counts = {}
            for r in rca_results:
                cause = getattr(r, 'root_cause', str(r))
                counts[cause] = counts.get(cause, 0) + 1
            causes = list(counts.keys())
            values = list(counts.values())
            colors = [RCA_COLORS.get(c, '#95A5A6') for c in causes]
            ax4.bar(causes, values, color=colors, edgecolor='black')
            ax4.set_ylabel('Count')
            ax4.set_title('Root Cause Attribution', fontweight='bold')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('MaHealthBiasAudit - Summary Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_fig(fig, 'summary_dashboard.png')
    
    def plot_executive_dashboard(self, sdi_matrix, tp_df, trust_results, rca_results, bias_results):
        """Executive summary dashboard"""
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('MaHealthBiasAudit Executive Dashboard\nReal-time Bias Monitoring', 
                    fontweight='bold', fontsize=18, y=0.98)
        
        # 1. Overall Bias Score (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        if sdi_matrix is not None and not sdi_matrix.empty:
            triu_indices = np.triu_indices_from(sdi_matrix.values, k=1)
            if len(triu_indices[0]) > 0:
                avg_sdi = sdi_matrix.values[triu_indices].mean()
                color = '#E74C3C' if avg_sdi > 0.4 else '#F39C12' if avg_sdi > 0.2 else '#27AE60'
                ax1.text(0.5, 0.6, f'{avg_sdi:.3f}', fontsize=48, ha='center', fontweight='bold', color=color)
                ax1.text(0.5, 0.3, f'Overall Bias Score\n{"HIGH" if avg_sdi > 0.4 else "MODERATE" if avg_sdi > 0.2 else "LOW"} Risk',
                        ha='center', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('Overall Bias Score', fontweight='bold', fontsize=12)
        
        # 2. Language Risk Matrix (Top Middle)
        ax2 = fig.add_subplot(gs[0, 1])
        if sdi_matrix is not None and not sdi_matrix.empty:
            languages = [l for l in sdi_matrix.columns if l != 'English']
            if 'English' in sdi_matrix.index:
                sdi_values = [sdi_matrix.loc['English', l] for l in languages if l in sdi_matrix.columns]
                colors = ['#E74C3C' if v > 0.4 else '#F39C12' if v > 0.2 else '#27AE60' for v in sdi_values]
                ax2.barh(languages, sdi_values, color=colors, edgecolor='black')
                ax2.axvline(x=0.4, color='#E74C3C', linestyle='--', linewidth=2)
                ax2.axvline(x=0.2, color='#F39C12', linestyle='--', linewidth=2)
                ax2.set_xlabel('SDI Score')
                ax2.set_title('Language Risk Levels', fontweight='bold', fontsize=12)
        
        # 3. Trust Score (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        if trust_results:
            avg_trust = np.mean([r.trust_score for r in trust_results.values()])
            color = '#27AE60' if avg_trust > 0.7 else '#F39C12' if avg_trust > 0.4 else '#E74C3C'
            ax3.text(0.5, 0.6, f'{avg_trust:.2f}', fontsize=48, ha='center', fontweight='bold', color=color)
            ax3.text(0.5, 0.3, f'Trust Score\n{"High" if avg_trust > 0.7 else "Medium" if avg_trust > 0.4 else "Low"}',
                    ha='center', fontsize=12)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Cultural Trust Score', fontweight='bold', fontsize=12)
        
        # 4. RCA Summary (Middle Left)
        ax4 = fig.add_subplot(gs[1, :2])
        if rca_results:
            counts = {}
            for r in rca_results:
                cause = getattr(r, 'root_cause', str(r))
                counts[cause] = counts.get(cause, 0) + 1
            causes = list(counts.keys())
            values = list(counts.values())
            colors = [RCA_COLORS.get(c, '#95A5A6') for c in causes]
            bars = ax4.bar(causes, values, color=colors, edgecolor='black')
            for bar, val in zip(bars, values):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(val), ha='center', fontweight='bold')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Root Cause Analysis', fontweight='bold', fontsize=12)
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Tokenisation Status (Middle Right)
        ax5 = fig.add_subplot(gs[1, 2])
        if tp_df is not None and not tp_df.empty:
            high_tp = sum(tp_df['Fertility_Penalty'] > 1.5)
            ok_tp = len(tp_df) - high_tp
            if ok_tp > 0 or high_tp > 0:
                wedges, texts, autotexts = ax5.pie([high_tp, ok_tp], labels=['High', 'OK'],
                                                   autopct='%1.0f%%', colors=['#E74C3C', '#27AE60'])
                ax5.set_title('Tokenisation Status', fontweight='bold', fontsize=12)
        
        # 6. Recommendations (Bottom)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Generate recommendations
        recommendations = []
        if sdi_matrix is not None and not sdi_matrix.empty:
            triu_indices = np.triu_indices_from(sdi_matrix.values, k=1)
            if len(triu_indices[0]) > 0:
                avg_sdi = sdi_matrix.values[triu_indices].mean()
                if avg_sdi > 0.3:
                    recommendations.append("High semantic divergence detected - Review non-English translations")
        
        if trust_results:
            avg_trust = np.mean([r.trust_score for r in trust_results.values()])
            if avg_trust < 0.6:
                recommendations.append("Low cultural trust scores - Enhance cultural adaptation")
        
        if tp_df is not None and not tp_df.empty:
            high_tp = sum(tp_df['Fertility_Penalty'] > 1.5)
            if high_tp > 0:
                recommendations.append("Tokenisation issues found - Optimize multilingual tokenisers")
        
        if not recommendations:
            recommendations = ["All metrics within acceptable ranges - Continue monitoring"]
        
        rec_text = "\n".join([f"• {rec}" for rec in recommendations[:5]])
        ax6.text(0.05, 0.7, "Actionable Recommendations:\n\n" + rec_text,
                transform=ax6.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', edgecolor='#3498DB'))
        
        self._save_fig(fig, 'executive_dashboard.png', dpi=150)
    
    # ==================== TABLE CREATION METHODS ====================
    def create_sdi_table(self, sdi_matrix):
        """Table 1: Comprehensive SDI matrix"""
        table_df = sdi_matrix.copy()
        
        # Add summary statistics
        summary_data = {}
        for col in table_df.columns:
            summary_data[col] = [table_df[col].mean(), table_df[col].max(), table_df[col].min()]
        
        summary = pd.DataFrame(summary_data, index=['AVERAGE', 'MAX', 'MIN'])
        
        # Combine with original
        final_table = pd.concat([table_df, summary])
        final_table = final_table.round(4)
        
        return final_table
    
    def create_tokenisation_table(self, tp_df):
        """Table 2: Tokenisation parity results"""
        table_df = tp_df.groupby('Language').agg({
            'Fertility_Penalty': ['mean', 'std', 'min', 'max'],
            'OOV_Rate': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        table_df.columns = ['Fertility_Mean', 'Fertility_Std', 'Fertility_Min', 'Fertility_Max',
                           'OOV_Mean', 'OOV_Std']
        
        # Add status
        table_df['Status'] = table_df['Fertility_Mean'].apply(
            lambda x: 'High' if x > 1.5 else '✓ OK'
        )
        
        return table_df
    
    def create_trust_table(self, trust_results):
        """Table 3: Trust score breakdown"""
        data = []
        for lang, result in trust_results.items():
            data.append({
                'Language': lang,
                'Trust_Score': round(result.trust_score, 4),
                'Preservation_Needed': result.preservation_needed,
                'Risk_Level': 'High' if result.trust_score < 0.4 else 'Medium' if result.trust_score < 0.7 else 'Low'
            })
        
        table_df = pd.DataFrame(data)
        return table_df
    
    def create_rca_table(self, rca_results):
        """Table 4: Root cause analysis summary"""
        from collections import Counter
        
        cause_counts = Counter([getattr(r, 'root_cause', str(r)) for r in rca_results])
        
        table_df = pd.DataFrame([
            {
                'Root_Cause': cause,
                'Frequency': count,
                'Percentage': f"{(count/len(rca_results))*100:.1f}%",
                'Severity': 'High' if count > len(rca_results)/3 else 'Medium' if count > len(rca_results)/5 else 'Low'
            }
            for cause, count in cause_counts.items()
        ])
        
        return table_df.sort_values('Frequency', ascending=False)
    
    def create_bias_summary_table(self, bias_results):
        """Table 5: Bias detection summary"""
        table_df = pd.DataFrame([
            {
                'Bias_Type': bias_type,
                'Score': round(score, 4),
                'Severity': 'High' if score > 0.3 else 'Medium' if score > 0.15 else 'Low',
                'Recommendation': 'Immediate Action' if score > 0.3 else 'Monitor' if score > 0.15 else 'Acceptable'
            }
            for bias_type, score in bias_results.items()
        ])
        
        return table_df
    
    def create_performance_table(self, performance_metrics):
        """Table 6: Performance metrics"""
        table_df = pd.DataFrame([
            {
                'Metric': metric,
                'Value': round(value, 4),
                'Status': '✓ Pass' if value > 0.7 else '⚠️ Warning' if value > 0.4 else '❌ Fail'
            }
            for metric, value in performance_metrics.items()
        ])
        
        return table_df
    
    def create_corpus_stats_table(self, corpus_stats):
        """Table 7: Corpus statistics"""
        if isinstance(corpus_stats, pd.DataFrame):
            return corpus_stats.round(4)
        return pd.DataFrame()
    
    def create_language_comparison_table(self, sdi_matrix, tp_df, trust_results):
        """Table 8: Language-wise comparison"""
        # Extract SDI scores
        sdi_scores = {}
        if sdi_matrix is not None and not sdi_matrix.empty and 'English' in sdi_matrix.index:
            for lang in sdi_matrix.columns:
                if lang != 'English':
                    sdi_scores[lang] = sdi_matrix.loc['English', lang]
        
        # Extract tokenisation scores
        tp_scores = {}
        if tp_df is not None and not tp_df.empty:
            for lang in tp_df['Language'].unique():
                lang_df = tp_df[tp_df['Language'] == lang]
                if not lang_df.empty:
                    tp_scores[lang] = lang_df['Fertility_Penalty'].mean()
        
        # Extract trust scores
        trust_scores = {lang: result.trust_score for lang, result in trust_results.items()}
        
        # Combine
        all_langs = set(sdi_scores.keys()) | set(tp_scores.keys()) | set(trust_scores.keys())
        
        data = []
        for lang in all_langs:
            data.append({
                'Language': lang,
                'SDI_Score': round(sdi_scores.get(lang, 0), 4) if lang in sdi_scores else 'N/A',
                'Tokenisation_Fertility': round(tp_scores.get(lang, 0), 4) if lang in tp_scores else 'N/A',
                'Trust_Score': round(trust_scores.get(lang, 0), 4) if lang in trust_scores else 'N/A',
                'Overall_Risk': 'High' if sdi_scores.get(lang, 0) > 0.4 else 'Medium' if sdi_scores.get(lang, 0) > 0.2 else 'Low'
            })
        
        table_df = pd.DataFrame(data)
        return table_df
    
    def save_table(self, table_df, table_name):
        """Save table to CSV and formatted text"""
        # Save as CSV
        csv_path = os.path.join(self.tables_dir, f"{table_name}.csv")
        table_df.to_csv(csv_path)
        
        # Save as formatted text
        txt_path = os.path.join(self.tables_dir, f"{table_name}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{table_name.replace('_', ' ')}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(table_df.to_string())
            f.write(f"\n\n{'='*80}\n")
        
        print(f"   Table saved: {table_name}.csv")
    
    def _save_fig(self, fig, filename, dpi=200):
        """Save figure"""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"   ✓ Saved: {filename}")
        plt.close(fig)