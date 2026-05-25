# benchmark/visualizer.py

"""
Visualization utilities for benchmark results.

Generates comprehensive plots:
1. Latency over time
2. Source distribution
3. Anchor evolution
4. Cache utilization
5. Hit rate comparison
6. Learning curves
7. Gate accuracy curve (Phase 5)
8. Coverage curve (Phase 5)
9. nDCG@10 over time (Phase 5)
10. MRR@10 over time (Phase 5)
11. Local hit rate over time (Phase 5)

Author: Saberzerker
Date: 2026-04-23 (Phase 5: added convergence plots)
"""

from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from benchmark.benchmark_config import config as benchmark_config

sns.set_style("whitegrid")


class BenchmarkVisualizer:
    """
    Creates visualizations for benchmark results.
    """
    
    def __init__(self, test_mode: str):
        """
        Initialize visualizer.
        
        Args:
            test_mode: "quick" or "full"
        """
        self.test_mode = test_mode
        self.output_dir = benchmark_config.RESULTS_DIR / test_mode
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_plots(self, results: Dict):
        """
        Generate all visualization plots.
        
        Args:
            results: Benchmark results dictionary
        """
        if not benchmark_config.SAVE_PLOTS:
            return
        
        print("\n[PLOTS] Generating visualizations...")
        
        df = pd.DataFrame(results['per_query'])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Latency over time
        self._plot_latency_over_time(fig.add_subplot(gs[0, :2]), df)
        
        # Plot 2: Source distribution (pie)
        self._plot_source_distribution(fig.add_subplot(gs[0, 2]), df)
        
        # Plot 3: Anchor evolution
        if len(results['anchor_snapshots']) > 0:
            anchor_df = pd.DataFrame(results['anchor_snapshots'])
            self._plot_anchor_evolution(fig.add_subplot(gs[1, :]), anchor_df)
        
        # Plot 4: Cache fill rate
        if len(results['cache_snapshots']) > 0:
            cache_df = pd.DataFrame(results['cache_snapshots'])
            self._plot_cache_fill_rate(fig.add_subplot(gs[2, 0]), cache_df)
            
            # Plot 5: Cache weight
            self._plot_cache_weight(fig.add_subplot(gs[2, 1]), cache_df)
        
        # Plot 6: Latency distribution
        self._plot_latency_distribution(fig.add_subplot(gs[2, 2]), df)
        
        # Save
        output_file = self.output_dir / "benchmark_plots.png"
        plt.savefig(output_file, dpi=benchmark_config.PLOT_DPI, bbox_inches='tight')
        print(f"[PLOTS] [OK] Saved to {output_file.name}")
        plt.close()
        
        # Generate per-query-type plots if applicable
        if 'query_type' in df.columns and df['query_type'].iloc[0] != 'unknown':
            self._generate_per_type_plots(df)

        # ═══════════════════════════════════════════════════════════════════════
        # Phase 5: Convergence plots (two separate curves)
        # Per THEORY.md §20: gate accuracy curve (fast, converges by Q20-50)
        # and coverage curve (slow, meaningful by vector 500+). Conflating
        # them weakens the contribution.
        # ═══════════════════════════════════════════════════════════════════════
        self._generate_convergence_plots(results, df)

    def _plot_latency_over_time(self, ax, df):
        """Plot latency over time with moving average."""
        ax.plot(df['query_num'], df['latency_ms'], 
                alpha=0.3, linewidth=0.5, color='gray', label='Raw')
        
        window = 20 if self.test_mode == "quick" else 50
        if len(df) >= window:
            rolling = df['latency_ms'].rolling(window=window).mean()
            ax.plot(df['query_num'], rolling, 
                   linewidth=2, color='#e74c3c', label=f'Moving avg ({window})')
        
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title('Latency Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_source_distribution(self, ax, df):
        """Plot source distribution as pie chart."""
        source_counts = df['source'].value_counts()
        colors = {
            'tier1_permanent': '#e74c3c',
            'tier2_dynamic': '#2ecc71',
            'tier3_cloud': '#3498db',
            'offline_fallback': '#95a5a6'
        }
        
        pie_colors = [colors.get(s, '#999') for s in source_counts.index]
        
        ax.pie(source_counts.values, labels=source_counts.index,
              autopct='%1.1f%%', colors=pie_colors, startangle=90)
        ax.set_title('Query Source Distribution', fontsize=12, fontweight='bold')
    
    def _plot_anchor_evolution(self, ax, anchor_df):
        """Plot anchor evolution as stacked area chart."""
        ax.fill_between(anchor_df['query_num'], 0, 
                        anchor_df.get('weak_anchors', 0),
                        label='Weak', alpha=0.7, color='#e74c3c')
        
        ax.fill_between(anchor_df['query_num'], 
                        anchor_df.get('weak_anchors', 0),
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0),
                        label='Medium', alpha=0.7, color='#f39c12')
        
        ax.fill_between(anchor_df['query_num'],
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0),
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0) + 
                        anchor_df.get('strong_anchors', 0),
                        label='Strong', alpha=0.7, color='#2ecc71')
        
        ax.fill_between(anchor_df['query_num'],
                        anchor_df.get('weak_anchors', 0) + anchor_df.get('medium_anchors', 0) + 
                        anchor_df.get('strong_anchors', 0),
                        anchor_df.get('total_anchors', 0),
                        label='Permanent', alpha=0.7, color='#3498db')
        
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Anchor Count', fontsize=11)
        ax.set_title('Anchor System Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_cache_fill_rate(self, ax, cache_df):
        """Plot cache fill rate over time."""
        ax.plot(cache_df['query_num'], cache_df.get('fill_rate', 0),
               linewidth=2, color='#3498db')
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Capacity')
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Fill Rate (%)', fontsize=11)
        ax.set_title('Cache Fill Rate', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_cache_weight(self, ax, cache_df):
        """Plot average cache weight over time."""
        ax.plot(cache_df['query_num'], cache_df.get('avg_weight', 0),
               linewidth=2, color='#2ecc71')
        ax.set_xlabel('Query Number', fontsize=11)
        ax.set_ylabel('Average Weight', fontsize=11)
        ax.set_title('Cache Vector Quality', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_latency_distribution(self, ax, df):
        """Plot latency distribution histogram."""
        ax.hist(df['latency_ms'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        mean_lat = df['latency_ms'].mean()
        ax.axvline(mean_lat, color='red', linestyle='--',
                  label=f'Mean: {mean_lat:.1f}ms', linewidth=2)
        ax.set_xlabel('Latency (ms)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Latency Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    def _generate_per_type_plots(self, df):
        """Generate separate plots for each query type."""
        print("[PLOTS] Generating per-query-type plots...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, qtype in enumerate(['in_dataset', 'edge_case', 'out_of_distribution']):
            type_df = df[df['query_type'] == qtype]
            
            if len(type_df) == 0:
                continue
            
            ax = axes[idx]
            
            # Calculate rolling hit rate
            is_local = type_df['source'].isin(['tier1_permanent', 'tier2_dynamic']).astype(int)
            window = 20
            if len(type_df) >= window:
                rolling_rate = is_local.rolling(window=window).mean() * 100
                ax.plot(type_df['query_num'], rolling_rate, linewidth=2)
            
            ax.set_xlabel('Query Number', fontsize=11)
            ax.set_ylabel('Local Hit Rate (%)', fontsize=11)
            ax.set_title(qtype.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "per_type_learning_curves.png"
        plt.savefig(output_file, dpi=benchmark_config.PLOT_DPI)
        print(f"[PLOTS] [OK] Saved to {output_file.name}")
        plt.close()

    def _generate_convergence_plots(self, results: Dict, df: pd.DataFrame):
        """Phase 5: Generate two convergence curves + quality metrics over time.

        Per THEORY.md §20: Two separate convergence curves are needed:
        1. Gate accuracy curve (fast, converges by Q20-50)
        2. Coverage curve (slow, meaningful by vector 500+)

        Plus quality metrics: nDCG@10, MRR@10, local hit rate over time.
        """
        print("[PLOTS] Generating Phase 5 convergence plots...")

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Phase 5: Hybrid Gate Convergence & Quality Metrics', fontsize=14, fontweight='bold')

        # ═══════════════════════════════════════════════════════════════════════
        # Plot 1: Gate accuracy curve (rolling precision of local serving)
        # ═══════════════════════════════════════════════════════════════════════
        ax1 = axes[0, 0]
        if 'gate_signal' in df.columns and 'source' in df.columns:
            # Local serving = tier1 or tier2
            is_local = df['source'].isin(['tier1_permanent', 'tier2_dynamic'])
            # Gate accuracy: fraction of local serves that were correct
            # (we approximate by local hit rate as proxy)
            window = 50
            if len(df) >= window:
                rolling_local = is_local.rolling(window=window).mean() * 100
                ax1.plot(df['query_num'], rolling_local, linewidth=2, color='#2ecc71', label='Local hit rate')
                ax1.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Target (70%)')
        ax1.set_xlabel('Query Number', fontsize=11)
        ax1.set_ylabel('Local Hit Rate (%)', fontsize=11)
        ax1.set_title('Gate Accuracy Curve\n(rolling local hit rate)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # ═══════════════════════════════════════════════════════════════════════
        # Plot 2: Coverage curve (Tier 2 fill rate over time)
        # ═══════════════════════════════════════════════════════════════════════
        ax2 = axes[0, 1]
        if len(results.get('cache_snapshots', [])) > 0:
            cache_df = pd.DataFrame(results['cache_snapshots'])
            ax2.plot(cache_df['query_num'], cache_df.get('current_size', 0),
                    linewidth=2, color='#3498db', label='Tier 2 size')
            if 'capacity' in cache_df.columns:
                ax2.axhline(y=cache_df['capacity'].iloc[0], color='red', linestyle='--',
                           alpha=0.5, label=f'Capacity ({int(cache_df["capacity"].iloc[0])})')
        ax2.set_xlabel('Query Number', fontsize=11)
        ax2.set_ylabel('Vectors in Tier 2', fontsize=11)
        ax2.set_title('Coverage Curve\n(Tier 2 fill over time)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ═══════════════════════════════════════════════════════════════════════
        # Plot 3: Gate signal distribution
        # ═══════════════════════════════════════════════════════════════════════
        ax3 = axes[0, 2]
        if 'gate_signal' in df.columns:
            gate_valid = df['gate_signal'].dropna()
            if len(gate_valid) > 0:
                local_mask = df['source'].isin(['tier1_permanent', 'tier2_dynamic'])
                cloud_mask = ~local_mask

                local_gate = df.loc[local_mask, 'gate_signal'].dropna()
                cloud_gate = df.loc[cloud_mask, 'gate_signal'].dropna()

                if len(local_gate) > 0:
                    ax3.hist(local_gate, bins=30, alpha=0.7,
                            color='#2ecc71', label='Served locally', edgecolor='black')
                if len(cloud_gate) > 0:
                    ax3.hist(cloud_gate, bins=30, alpha=0.7,
                            color='#3498db', label='Fell to cloud', edgecolor='black')

                if 'gate_threshold' in df.columns:
                    threshold_vals = df['gate_threshold'].dropna()
                    if len(threshold_vals) > 0:
                        threshold = threshold_vals.iloc[-1]
                        ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                                   label=f'Threshold ({threshold:.2f})')
            else:
                ax3.text(0.5, 0.5, 'No gate signal data\n(cloud_only mode)', transform=ax3.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
        else:
            ax3.text(0.5, 0.5, 'No gate signal data\n(cloud_only mode)', transform=ax3.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
        ax3.set_xlabel('Gate Signal', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Gate Signal Distribution\n(BM25+cosine hybrid)', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ═══════════════════════════════════════════════════════════════════════
        # Plot 4: nDCG@10 over time (if labeled data available)
        # ═══════════════════════════════════════════════════════════════════════
        ax4 = axes[1, 0]
        if 'ndcg_at_k' in df.columns:
            labeled = df.dropna(subset=['ndcg_at_k'])
            if len(labeled) > 0:
                window = min(50, max(5, len(labeled) // 10))
                rolling_ndcg = labeled['ndcg_at_k'].rolling(window=window).mean()
                ax4.plot(labeled['query_num'], rolling_ndcg, linewidth=2, color='#9b59b6')
                ax4.scatter(labeled['query_num'], labeled['ndcg_at_k'], alpha=0.15, s=10, color='#9b59b6')
                ax4.set_xlabel('Query Number', fontsize=11)
                ax4.set_ylabel('nDCG@10', fontsize=11)
                ax4.set_title('nDCG@10 Over Time\n(rolling mean)', fontsize=12, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'No labeled data\nfor nDCG@10', transform=ax4.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax4.set_title('nDCG@10 Over Time', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # ═══════════════════════════════════════════════════════════════════════
        # Plot 5: MRR@10 over time (if labeled data available)
        # ═══════════════════════════════════════════════════════════════════════
        ax5 = axes[1, 1]
        if 'mrr_at_k' in df.columns:
            labeled = df.dropna(subset=['mrr_at_k'])
            if len(labeled) > 0:
                window = min(50, max(5, len(labeled) // 10))
                rolling_mrr = labeled['mrr_at_k'].rolling(window=window).mean()
                ax5.plot(labeled['query_num'], rolling_mrr, linewidth=2, color='#e67e22')
                ax5.scatter(labeled['query_num'], labeled['mrr_at_k'], alpha=0.15, s=10, color='#e67e22')
                ax5.set_xlabel('Query Number', fontsize=11)
                ax5.set_ylabel('MRR@10', fontsize=11)
                ax5.set_title('MRR@10 Over Time\n(rolling mean)', fontsize=12, fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No labeled data\nfor MRR@10', transform=ax5.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax5.set_title('MRR@10 Over Time', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # ═══════════════════════════════════════════════════════════════════════
        # Plot 6: Local hit rate over time (all queries)
        # ═══════════════════════════════════════════════════════════════════════
        ax6 = axes[1, 2]
        is_local = df['source'].isin(['tier1_permanent', 'tier2_dynamic']).astype(int)
        window = 50
        if len(df) >= window:
            rolling_hit = is_local.rolling(window=window).mean() * 100
            ax6.plot(df['query_num'], rolling_hit, linewidth=2, color='#2ecc71', label='Local hit rate')
            # Also show tier breakdown if possible
            is_tier2 = (df['source'] == 'tier2_dynamic').astype(int)
            is_tier1 = (df['source'] == 'tier1_permanent').astype(int)
            if len(df) >= window:
                rolling_t2 = is_tier2.rolling(window=window).mean() * 100
                rolling_t1 = is_tier1.rolling(window=window).mean() * 100
                ax6.plot(df['query_num'], rolling_t2, linewidth=1.5, color='#3498db',
                        linestyle='--', label='Tier 2 only')
                ax6.plot(df['query_num'], rolling_t1, linewidth=1.5, color='#e74c3c',
                        linestyle=':', label='Tier 1 only')
        ax6.set_xlabel('Query Number', fontsize=11)
        ax6.set_ylabel('Hit Rate (%)', fontsize=11)
        ax6.set_title('Local Hit Rate Over Time\n(rolling 50-query window)', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / "phase5_convergence_plots.png"
        plt.savefig(output_file, dpi=benchmark_config.PLOT_DPI, bbox_inches='tight')
        print(f"[PLOTS] [OK] Phase 5 convergence plots saved to {output_file.name}")
        plt.close()