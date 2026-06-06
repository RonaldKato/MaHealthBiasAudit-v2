"""
MaHealthBiasAudit - Visualization Dashboard
Saves all 18 visualizations as PNG files with clear labels
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import FIGURES_DIR, VIZ_HEIGHT, VIZ_WIDTH, COLOR_PALETTES, FIGURE_DPI
from utils import setup_logger


class VisualizationDashboard:
    """Generate and save comprehensive visualizations for bias audit"""
    
    def __init__(self):
        self.logger = setup_logger('visualization')
        self.figures_dir = FIGURES_DIR
        self.setup_style()
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def setup_style(self):
        """Setup plotting style"""
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        sns.set_palette(COLOR_PALETTES['main'])
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['savefig.dpi'] = FIGURE_DPI
        plt.rcParams['savefig.bbox'] = 'tight'
    
    # ============================================================
    # VISUALIZATION 1: SDI Heatmap
    # ============================================================
    def save_sdi_heatmap(self, sdi_matrix: pd.DataFrame) -> None:
        """Visualization 1: Semantic Distance Index Heatmap"""
        if sdi_matrix.empty:
            self.logger.warning("SDI matrix empty, skipping heatmap")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(sdi_matrix.values, cmap='RdYlBu_r', vmin=0, vmax=0.8)
        
        ax.set_xticks(range(len(sdi_matrix.columns)))
        ax.set_yticks(range(len(sdi_matrix.index)))
        ax.set_xticklabels(sdi_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(sdi_matrix.index)
        ax.set_title('Visualization 1: Semantic Distance Index (SDI) Heatmap\n(Higher values = more bias)', fontsize=14, fontweight='bold')
        
        for i in range(len(sdi_matrix.index)):
            for j in range(len(sdi_matrix.columns)):
                ax.text(j, i, f"{sdi_matrix.values[i, j]:.3f}",
                       ha="center", va="center", 
                       color="white" if sdi_matrix.values[i, j] > 0.5 else "black", fontsize=10)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SDI Score (0=identical, 1=completely different)', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '1_sdi_heatmap.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 1_sdi_heatmap.png")
    
    # ============================================================
    # VISUALIZATION 2: Response Length by Language
    # ============================================================
    def save_response_length_chart(self, length_stats: pd.DataFrame) -> None:
        """Visualization 2: Response Length Comparison by Language"""
        if length_stats.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(length_stats['Language'], length_stats['Mean'], 
                     yerr=length_stats['Std'], capsize=5,
                     color='steelblue', alpha=0.8, edgecolor='black')
        
        for bar, val in zip(bars, length_stats['Mean']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Average Words per Response', fontsize=12)
        ax.set_title('Visualization 2: Response Length by Language', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '2_response_length.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 2_response_length.png")
    
    # ============================================================
    # VISUALIZATION 3: Tokeniser Performance
    # ============================================================
    def save_tokeniser_performance_chart(self, tp_df: pd.DataFrame) -> None:
        """Visualization 3: Tokeniser Fertility Penalty by Language"""
        if tp_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        tokenisers = tp_df['Tokeniser'].unique()
        languages = tp_df['Language'].unique()
        x = np.arange(len(languages))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (tokeniser, color) in enumerate(zip(tokenisers, colors)):
            tokeniser_data = tp_df[tp_df['Tokeniser'] == tokeniser]
            values = []
            for lang in languages:
                val = tokeniser_data[tokeniser_data['Language'] == lang]['Fertility_Penalty'].values
                values.append(val[0] if len(val) > 0 else 0)
            bars = ax.bar(x + i*width, values, width, label=tokeniser, color=color, alpha=0.8)
            
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.axhline(y=1.5, color='red', linestyle='--', linewidth=2, label='Warning Threshold')
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Fertility Penalty', fontsize=12)
        ax.set_title('Visualization 3: Tokeniser Fertility Penalty by Language\n(Lower is better)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(languages)
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '3_tokeniser_performance.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 3_tokeniser_performance.png")
    
    # ============================================================
    # VISUALIZATION 4: Vocabulary Richness
    # ============================================================
    def save_vocabulary_richness_chart(self, vocab_stats: pd.DataFrame) -> None:
        """Visualization 4: Vocabulary Richness and Lexical Diversity"""
        if vocab_stats.empty:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(vocab_stats['Language']))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, vocab_stats['Vocabulary_Richness_Mean'], width, 
                      label='Vocabulary Richness (TTR)', color='steelblue', alpha=0.8,
                      yerr=vocab_stats['Vocabulary_Richness_Std'], capsize=3)
        bars2 = ax.bar(x + width/2, vocab_stats['Lexical_Diversity_Mean'], width,
                      label='Lexical Diversity (MATTR)', color='coral', alpha=0.8,
                      yerr=vocab_stats['Lexical_Diversity_Std'], capsize=3)
        
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Visualization 4: Vocabulary Richness and Lexical Diversity\n(Higher = richer vocabulary)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(vocab_stats['Language'])
        ax.legend(loc='lower right')
        ax.set_ylim(0, 0.8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '4_vocabulary_richness.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 4_vocabulary_richness.png")
    
    # ============================================================
    # VISUALIZATION 5: SDI Ranking
    # ============================================================
    def save_sdi_ranking_chart(self, sdi_ranking: Dict[str, float]) -> None:
        """Visualization 5: SDI Ranking Bar Chart"""
        if not sdi_ranking:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        languages = list(sdi_ranking.keys())
        scores = list(sdi_ranking.values())
        colors = ['#d62728' if s > 0.4 else '#ff7f0e' if s > 0.2 else '#2ca02c' for s in scores]
        
        bars = ax.barh(languages, scores, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        ax.axvline(x=0.4, color='red', linestyle='--', linewidth=2, label='High Bias Threshold')
        ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=2, label='Moderate Bias Threshold')
        ax.set_xlabel('SDI Score (vs English)', fontsize=12)
        ax.set_ylabel('Language', fontsize=12)
        ax.set_title('Visualization 5: Semantic Distance Index (SDI) Ranking\n(Higher = more bias compared to English)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '5_sdi_ranking.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 5_sdi_ranking.png")
    
    # ============================================================
    # VISUALIZATION 6: Root Cause Pie Chart
    # ============================================================
    def save_root_cause_pie_chart(self, rca_counts: Dict[str, int]) -> None:
        """Visualization 6: Root Cause Distribution Pie Chart"""
        if not rca_counts:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        labels = list(rca_counts.keys())
        sizes = list(rca_counts.values())
        colors = COLOR_PALETTES['main'][:len(labels)]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                          colors=colors, startangle=90,
                                          textprops={'fontsize': 11})
        
        ax.set_title('Visualization 6: Root Cause Distribution\n(Reasons for cross-lingual bias)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '6_root_cause_pie.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 6_root_cause_pie.png")
    
    # ============================================================
    # VISUALIZATION 7: Trust-Aware Metrics
    # ============================================================
    def save_trust_metrics_chart(self, trust_results: List[Dict]) -> None:
        """Visualization 7: Trust-Aware Metrics by Language Pair"""
        if not trust_results:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        comparisons = [r.get('Comparison', 'Unknown')[:20] for r in trust_results]
        ratios = [r.get('Alignment_Ratio', 0) for r in trust_results]
        colors = ['green' if r > 0.8 else 'orange' if r > 0.6 else 'red' for r in ratios]
        
        bars = ax.bar(comparisons, ratios, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, ratio in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{ratio:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax.axhline(y=0.8, color='green', linestyle='--', linewidth=2, label='High Trust')
        ax.axhline(y=0.6, color='orange', linestyle='--', linewidth=2, label='Medium Trust')
        ax.set_xlabel('Language Comparison', fontsize=12)
        ax.set_ylabel('Alignment Ratio (vs English)', fontsize=12)
        ax.set_title('Visualization 7: Trust-Aware Metrics\n(Higher = better alignment with English)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '7_trust_metrics.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 7_trust_metrics.png")
    
    # ============================================================
    # VISUALIZATION 8: Flags Distribution
    # ============================================================
    def save_flags_distribution(self, flags: List[Dict]) -> None:
        """Visualization 8: Distribution of Bias Flags by Type"""
        if not flags:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        flag_counts = {}
        for flag in flags:
            flag_type = flag.get('Type', 'Unknown')
            flag_counts[flag_type] = flag_counts.get(flag_type, 0) + 1
        
        types = list(flag_counts.keys())
        counts = list(flag_counts.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
        
        bars = ax.bar(types, counts, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_xlabel('Bias Flag Type', fontsize=12)
        ax.set_ylabel('Number of Flags', fontsize=12)
        ax.set_title('Visualization 8: Distribution of Bias Flags by Type', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '8_flags_distribution.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 8_flags_distribution.png")
    
    # ============================================================
    # VISUALIZATION 9: Statistical Test Results Table
    # ============================================================
    def save_statistical_tests_table(self, test_results: List[Dict]) -> None:
        """Visualization 9: Statistical Test Results Table"""
        if not test_results:
            return
        
        fig, ax = plt.subplots(figsize=(14, len(test_results) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        headers = ['Comparison', 'Mean Diff', 'Mann-Whitney U', 'P-Value', 'Significant', 'Effect Size', 'Interpretation']
        data = []
        for r in test_results:
            data.append([
                r.get('Comparison', ''),
                f"{r.get('Mean_Diff', 0):.2f}",
                f"{r.get('Mann_Whitney_U', 0):.0f}",
                f"{r.get('P_Value', 1):.4f}",
                '✓' if r.get('Significant', False) else '✗',
                f"{r.get('Effect_Size_Cohens_d', 0):.3f}",
                r.get('Interpretation', '')[:50] + '...' if len(r.get('Interpretation', '')) > 50 else r.get('Interpretation', '')
            ])
        
        table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        for i, r in enumerate(test_results):
            if r.get('Significant', False):
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#ffcccc')
        
        ax.set_title('Visualization 9: Statistical Test Results (English vs Others)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '9_statistical_tests_table.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 9_statistical_tests_table.png")
    
    # ============================================================
    # VISUALIZATION 10: Category Summary Table
    # ============================================================
    def save_category_summary_table(self, category_stats: pd.DataFrame) -> None:
        """Visualization 10: Category-wise Summary Table"""
        if category_stats.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        try:
            pivot_table = category_stats.pivot(index='Category', columns='Language', values='Avg_Length').round(1)
            pivot_table.loc['Overall Average'] = pivot_table.mean()
            
            table = ax.table(cellText=pivot_table.values, 
                            rowLabels=pivot_table.index,
                            colLabels=pivot_table.columns,
                            loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            ax.set_title('Visualization 10: Category-wise Average Response Length by Language', fontsize=14, fontweight='bold', pad=20)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, '10_category_summary_table.png'), dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: 10_category_summary_table.png")
        except Exception as e:
            print(f"  ⚠ Could not save category summary table: {e}")
    
    # ============================================================
    # VISUALIZATION 11: Dataset Statistics Table
    # ============================================================
    def save_dataset_statistics_table(self, stats_df: pd.DataFrame) -> None:
        """Visualization 11: Dataset Statistics Table"""
        if stats_df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Check actual column names and map them
        # Expected columns: 'Language', 'Num_Answers', 'Avg_Length_Words', 'Std_Length_Words', 'Min_Length_Words', 'Max_Length_Words', 'Avg_Sentence_Length'
        # Also possible: 'Language', 'Count', 'Mean', 'Std', 'Min', 'Max'
        
        if 'Num_Answers' in stats_df.columns:
            headers = ['Language', 'Num Answers', 'Avg Length', 'Std Length', 'Min', 'Max', 'Avg Sentence Length']
            data = stats_df[['Language', 'Num_Answers', 'Avg_Length_Words', 'Std_Length_Words', 
                           'Min_Length_Words', 'Max_Length_Words', 'Avg_Sentence_Length']].values.tolist()
        elif 'Count' in stats_df.columns:
            headers = ['Language', 'Count', 'Mean', 'Std', 'Min', 'Max']
            data = stats_df[['Language', 'Count', 'Mean', 'Std', 'Min', 'Max']].values.tolist()
        else:
            # Fallback: use available columns
            available_cols = stats_df.columns.tolist()
            headers = available_cols
            data = stats_df.values.tolist()
        
        table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title('Visualization 11: Dataset Statistics by Language', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '11_dataset_statistics_table.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 11_dataset_statistics_table.png")
    
    # ============================================================
    # VISUALIZATION 12: Flag Details Table
    # ============================================================
    def save_flag_details_table(self, flags: List[Dict]) -> None:
        """Visualization 12: Flag Details Table"""
        if not flags:
            return
        
        fig, ax = plt.subplots(figsize=(14, min(len(flags) * 0.4 + 2, 12)))
        ax.axis('tight')
        ax.axis('off')
        
        headers = ['Type', 'Language/Comparison', 'Severity', 'Description', 'Recommendation']
        data = []
        for f in flags[:20]:
            data.append([
                f.get('Type', ''),
                f.get('Language', f.get('Comparison', '')),
                f.get('Severity', ''),
                f.get('Description', '')[:60] + '...' if len(f.get('Description', '')) > 60 else f.get('Description', ''),
                f.get('Recommendation', '')[:60] + '...' if len(f.get('Recommendation', '')) > 60 else f.get('Recommendation', '')
            ])
        
        table = ax.table(cellText=data, colLabels=headers, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        for i, f in enumerate(flags[:20]):
            severity = f.get('Severity', '')
            if severity == 'High':
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#ffcccc')
            elif severity == 'Moderate':
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#ffffcc')
        
        ax.set_title('Visualization 12: Bias Flag Details (Top 20)', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '12_flag_details_table.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 12_flag_details_table.png")
    
    # ============================================================
    # VISUALIZATION 13: Recommendations Table
    # ============================================================
    def save_recommendations_table(self, recommendations: List[str]) -> None:
        """Visualization 13: Recommendations Table"""
        if not recommendations:
            return
        
        fig, ax = plt.subplots(figsize=(12, max(len(recommendations) * 0.5 + 2, 4)))
        ax.axis('tight')
        ax.axis('off')
        
        data = [[f"{i+1}. {rec}"] for i, rec in enumerate(recommendations)]
        
        table = ax.table(cellText=data, colLabels=['Recommendations'], loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        
        ax.set_title('Visualization 13: Actionable Recommendations', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '13_recommendations_table.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 13_recommendations_table.png")
    
    # ============================================================
    # VISUALIZATION 14: Response Length Violin Plot
    # ============================================================
    def save_violin_plot(self, length_data: Dict[str, List[int]]) -> None:
        """Visualization 14: Violin Plot of Response Lengths"""
        if not length_data:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [length_data[lang] for lang in length_data.keys()]
        labels = list(length_data.keys())
        
        parts = ax.violinplot(data, positions=range(len(labels)), showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLOR_PALETTES['main'][i % len(COLOR_PALETTES['main'])])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Response Length (words)', fontsize=12)
        ax.set_title('Visualization 14: Distribution of Response Lengths by Language', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '14_violin_plot.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 14_violin_plot.png")
    
    # ============================================================
    # VISUALIZATION 15: Overall Bias Gauge
    # ============================================================
    def save_bias_gauge(self, avg_sdi: float, bias_level: str, total_flags: int) -> None:
        """Visualization 15: Overall Bias Gauge Chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = {'HIGH': '#d62728', 'MODERATE': '#ff7f0e', 'LOW': '#2ca02c'}
        color = colors.get(bias_level, '#1f77b4')
        
        theta = np.linspace(0, np.pi, 100)
        value_rad = (avg_sdi * np.pi)
        theta_value = np.linspace(0, value_rad, 100)
        
        ax.plot(np.cos(theta), np.sin(theta), color='lightgray', linewidth=20)
        ax.plot(np.cos(theta_value), np.sin(theta_value), color=color, linewidth=20)
        
        needle_angle = value_rad
        ax.arrow(0, 0, 0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle),
                head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax.text(0, -0.3, f'SDI Score: {avg_sdi:.3f}', ha='center', va='center', fontsize=16, fontweight='bold')
        ax.text(0, -0.5, f'Bias Level: {bias_level}', ha='center', va='center', fontsize=14, color=color, fontweight='bold')
        ax.text(0, -0.7, f'Total Issues: {total_flags}', ha='center', va='center', fontsize=12)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Visualization 15: Overall Bias Assessment', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '15_bias_gauge.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 15_bias_gauge.png")
    
    # ============================================================
    # VISUALIZATION 16: Outliers Analysis
    # ============================================================
    def save_outliers_chart(self, outliers: List[Dict]) -> None:
        """Visualization 16: Outliers Distribution by Language"""
        if not outliers:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        outlier_counts = {}
        for o in outliers:
            lang = o.get('Language', 'Unknown')
            outlier_counts[lang] = outlier_counts.get(lang, 0) + 1
        
        languages = list(outlier_counts.keys())
        counts = list(outlier_counts.values())
        
        bars = ax.bar(languages, counts, color='coral', alpha=0.8, edgecolor='black')
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Number of Outliers', fontsize=12)
        ax.set_title('Visualization 16: Response Outliers by Language', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '16_outliers_chart.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 16_outliers_chart.png")
    
    # ============================================================
    # VISUALIZATION 17: N-gram Analysis Heatmap
    # ============================================================
    def save_ngram_heatmap(self, ngram_data: Dict[str, List[Tuple[str, int]]], n: int = 2) -> None:
        """Visualization 17: N-gram Analysis Heatmap"""
        if not ngram_data:
            return
        
        all_ngrams = set()
        for ngrams in ngram_data.values():
            for ng, _ in ngrams[:10]:
                all_ngrams.add(ng)
        
        all_ngrams = list(all_ngrams)[:10]
        
        if not all_ngrams:
            return
        
        matrix = []
        languages = list(ngram_data.keys())
        for lang in languages:
            ng_dict = dict(ngram_data.get(lang, []))
            row = [ng_dict.get(ng, 0) for ng in all_ngrams]
            matrix.append(row)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        ax.set_xticks(range(len(all_ngrams)))
        ax.set_yticks(range(len(languages)))
        ax.set_xticklabels(all_ngrams, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(languages)
        ax.set_title(f'Visualization 17: Top {n}-gram Frequency by Language', fontsize=14, fontweight='bold')
        
        for i in range(len(languages)):
            for j in range(len(all_ngrams)):
                if matrix[i][j] > 0:
                    ax.text(j, i, str(matrix[i][j]), ha='center', va='center', fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '17_ngram_heatmap.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 17_ngram_heatmap.png")
    
    # ============================================================
    # VISUALIZATION 18: Executive Dashboard
    # ============================================================
    def save_executive_dashboard(self, report: Dict[str, Any]) -> None:
        """Visualization 18: Executive Summary Dashboard"""
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('MaHealthBiasAudit: Executive Summary Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        # Panel 1: Key Metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        metrics_text = f"""
        KEY METRICS
        ─────────────────
        Average SDI:     {report['key_metrics']['average_sdi']:.4f}
        Bias Level:      {report['key_metrics']['bias_level']}
        Total Flags:     {report['key_metrics']['total_flags']}
        Languages:       {', '.join(report['languages'])}
        Total Answers:   {report['total_answers']}
        """
        ax1.text(0.1, 0.5, metrics_text, transform=ax1.transAxes, fontsize=12, verticalalignment='center', fontfamily='monospace')
        ax1.set_title('Key Performance Indicators', fontsize=12, fontweight='bold')
        
        # Panel 2: SDI Ranking
        ax2 = fig.add_subplot(gs[0, 1])
        sdi_ranking = report.get('sdi_ranking', {})
        if sdi_ranking:
            languages = list(sdi_ranking.keys())
            scores = list(sdi_ranking.values())
            colors = ['red' if s > 0.4 else 'orange' if s > 0.2 else 'green' for s in scores]
            ax2.barh(languages, scores, color=colors)
            ax2.set_xlabel('SDI Score')
            ax2.set_title('SDI Ranking (vs English)')
            ax2.axvline(x=0.4, color='red', linestyle='--', alpha=0.5)
            ax2.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5)
        
        # Panel 3: Bias Level Gauge
        ax3 = fig.add_subplot(gs[0, 2])
        avg_sdi = report['key_metrics']['average_sdi']
        bias_level = report['key_metrics']['bias_level']
        colors = {'HIGH': 'red', 'MODERATE': 'orange', 'LOW': 'green'}
        
        theta = np.linspace(0, np.pi, 100)
        ax3.plot(np.cos(theta), np.sin(theta), color='lightgray', linewidth=15)
        value_rad = avg_sdi * np.pi
        theta_value = np.linspace(0, value_rad, 100)
        ax3.plot(np.cos(theta_value), np.sin(theta_value), color=colors.get(bias_level, 'blue'), linewidth=15)
        ax3.arrow(0, 0, 0.7 * np.cos(value_rad), 0.7 * np.sin(value_rad),
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        ax3.text(0, -0.3, f'{avg_sdi:.3f}', ha='center', va='center', fontsize=14, fontweight='bold')
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-1.2, 1.2)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title(f'Overall Bias: {bias_level}', fontsize=12, fontweight='bold', color=colors.get(bias_level, 'blue'))
        
        # Panel 4: Root Cause Distribution
        ax4 = fig.add_subplot(gs[1, 0])
        rca_counts = report.get('rca_distribution', {})
        if rca_counts:
            labels = list(rca_counts.keys())
            sizes = list(rca_counts.values())
            ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Root Cause Distribution')
        
        # Panel 5: Recommendations
        ax5 = fig.add_subplot(gs[1, 1:3])
        ax5.axis('off')
        recommendations = report.get('recommendations', [])
        rec_text = "RECOMMENDATIONS\n" + "─" * 40 + "\n"
        for i, rec in enumerate(recommendations[:6], 1):
            rec_text += f"{i}. {rec}\n\n"
        ax5.text(0.05, 0.95, rec_text, transform=ax5.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax5.set_title('Actionable Recommendations', fontsize=12, fontweight='bold')
        
        # Panel 6: Flag Summary
        ax6 = fig.add_subplot(gs[2, 0])
        flags = report.get('flags', [])
        flag_types = {}
        for f in flags:
            ft = f.get('Type', 'Unknown')
            flag_types[ft] = flag_types.get(ft, 0) + 1
        if flag_types:
            ax6.bar(list(flag_types.keys()), list(flag_types.values()), color='steelblue')
            ax6.set_xlabel('Flag Type')
            ax6.set_ylabel('Count')
            ax6.set_title('Flag Distribution')
            plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        
        # Panel 7: Footer
        ax7 = fig.add_subplot(gs[2, 1:3])
        ax7.axis('off')
        footer_text = f"""
        Report Generated: {report.get('timestamp', 'N/A')}
        Experiment: {report.get('experiment', 'MaHealthBiasAudit')}
        Output Directory: {FIGURES_DIR}
        """
        ax7.text(0.1, 0.5, footer_text, transform=ax7.transAxes, fontsize=10, verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '18_executive_dashboard.png'), dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: 18_executive_dashboard.png")
    
    # ============================================================
    # Main method to save all visualizations
    # ============================================================
    def save_all_visualizations(self, 
                                sdi_matrix: pd.DataFrame,
                                length_stats: pd.DataFrame,
                                tp_df: pd.DataFrame,
                                trust_results: List[Dict],
                                rca_counts: Dict[str, int],
                                sdi_ranking: Dict[str, float],
                                flags: List[Dict],
                                length_data: Dict[str, List[int]],
                                avg_sdi: float,
                                bias_level: str,
                                total_flags: int,
                                test_results: List[Dict] = None,
                                category_stats: pd.DataFrame = None,
                                stats_df: pd.DataFrame = None,
                                recommendations: List[str] = None,
                                outliers: List[Dict] = None,
                                ngram_data: Dict = None,
                                report: Dict = None,
                                vocab_stats: pd.DataFrame = None) -> None:
        """Save all 18 visualizations as PNG files"""
        
        print("\n" + "=" * 70)
        print("SAVING ALL 18 VISUALIZATIONS")
        print("=" * 70)
        print(f"Output directory: {self.figures_dir}\n")
        
        print("Generating visualizations...")
        
        # 1. SDI Heatmap
        self.save_sdi_heatmap(sdi_matrix)
        
        # 2. Response Length Chart
        if length_stats is not None and not length_stats.empty:
            self.save_response_length_chart(length_stats)
        
        # 3. Tokeniser Performance
        if tp_df is not None and not tp_df.empty:
            self.save_tokeniser_performance_chart(tp_df)
        
        # 4. Vocabulary Richness
        if vocab_stats is not None and not vocab_stats.empty:
            self.save_vocabulary_richness_chart(vocab_stats)
        
        # 5. SDI Ranking
        if sdi_ranking:
            self.save_sdi_ranking_chart(sdi_ranking)
        
        # 6. Root Cause Pie Chart
        if rca_counts:
            self.save_root_cause_pie_chart(rca_counts)
        
        # 7. Trust Metrics
        if trust_results:
            self.save_trust_metrics_chart(trust_results)
        
        # 8. Flags Distribution
        if flags:
            self.save_flags_distribution(flags)
        
        # 9. Statistical Tests Table
        if test_results:
            self.save_statistical_tests_table(test_results)
        
        # 10. Category Summary Table
        if category_stats is not None and not category_stats.empty:
            self.save_category_summary_table(category_stats)
        
        # 11. Dataset Statistics Table
        if stats_df is not None and not stats_df.empty:
            self.save_dataset_statistics_table(stats_df)
        
        # 12. Flag Details Table
        if flags:
            self.save_flag_details_table(flags)
        
        # 13. Recommendations Table
        if recommendations:
            self.save_recommendations_table(recommendations)
        
        # 14. Violin Plot
        if length_data:
            self.save_violin_plot(length_data)
        
        # 15. Bias Gauge
        self.save_bias_gauge(avg_sdi, bias_level, total_flags)
        
        # 16. Outliers Chart
        if outliers:
            self.save_outliers_chart(outliers)
        
        # 17. N-gram Heatmap
        if ngram_data:
            self.save_ngram_heatmap(ngram_data, n=2)
        
        # 18. Executive Dashboard
        if report:
            self.save_executive_dashboard(report)
        
        print("\n" + "=" * 70)
        print(f"✓ All 18 visualizations saved to: {self.figures_dir}")
        print("=" * 70)