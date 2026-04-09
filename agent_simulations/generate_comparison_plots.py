#!/usr/bin/env python3
"""
Lightweight visualization generator for Optuna tuning results.
Loads existing JSON results and generates comparison plots.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    results_dir = Path('arrays/optuna_10k_scenario5')
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1
    
    # Load all results
    all_results = {}
    for agent_file in sorted(results_dir.glob('best_*.json')):
        agent_name = agent_file.stem.replace('best_', '')
        try:
            with open(agent_file) as f:
                result = json.load(f)
                # Normalize agent name
                if agent_name.lower() == 'normalthompsonsampling':
                    agent_name = 'NormalThompsonSampling'
                else:
                    agent_name = agent_name.capitalize() if agent_name.lower() in ['ucb', 'egreedy', 'softmax'] else agent_name
                all_results[agent_name] = result
                print(f"✓ Loaded {agent_name}: phase3_mean={result['metrics']['phase3_mean']:.2f}, phase3_std={result['metrics']['phase3_std']:.2f}")
        except Exception as e:
            print(f"✗ Failed to load {agent_file}: {e}")
    
    if not all_results:
        print("No results found")
        return 1
    
    print(f"\nGenerating visualizations for {len(all_results)} algorithms...")
    
    # === PLOT 1: Individual algorithm traces ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('All Algorithms - Scenario5 10k Runs (Pareto-Optimized)', fontsize=16, fontweight='bold')
    
    agents_list = list(all_results.keys())
    for idx, agent_name in enumerate(agents_list[:4]):
        ax = axes[idx // 2, idx % 2]
        
        # Load throughput array
        array_file = results_dir / f"{agent_name}_best_throughput.npy"
        if not array_file.exists():
            ax.text(0.5, 0.5, f'{agent_name}: No array file', ha='center', va='center')
            continue
        
        throughput = np.load(str(array_file))
        metrics = all_results[agent_name]['metrics']
        
        # Plot throughput
        kernel = np.ones(100) / 100
        smoothed = np.convolve(throughput, kernel, mode='valid')
        ax.plot(np.arange(len(smoothed)) + 50, smoothed, linewidth=2.5, color='navy', label='Smoothed (w=100)')
        
        # Phase markers
        ax.axvline(x=1000, color='green', linestyle='--', alpha=0.6, linewidth=2, label='End Phase 1')
        ax.axvline(x=5000, color='red', linestyle='--', alpha=0.6, linewidth=2, label='Start Phase 3')
        ax.axvspan(5000, 10000, alpha=0.1, color='red')
        
        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel('Throughput (Mbps)', fontsize=11)
        ax.set_title(f"{agent_name}\nMean(Ph3): {metrics['phase3_mean']:.2f} | Std(Ph3): {metrics['phase3_std']:.2f}",
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = results_dir / "all_algorithms_comparison.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # === PLOT 2: Metrics comparison bars ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Metrics Comparison (Best from Pareto Front)', fontsize=14, fontweight='bold')
    
    agents = list(all_results.keys())
    phase3_means = [all_results[a]['metrics']['phase3_mean'] for a in agents]
    phase3_stds = [all_results[a]['metrics']['phase3_std'] for a in agents]
    
    # Subplot 1: Mean throughput
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(agents)))
    bars = ax.bar(agents, phase3_means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Phase 3 Mean Throughput (5k-10k) ↑ Higher is Better', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, phase3_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{mean:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 2: Variance (lower is better)
    ax = axes[1]
    colors_std = ['#2ecc71' if std < 30 else '#f39c12' if std < 50 else '#e74c3c' for std in phase3_stds]
    bars = ax.bar(agents, phase3_stds, color=colors_std, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Standard Deviation (Mbps)', fontsize=11)
    ax.set_title('Phase 3 Variance ↓ Lower is Better (CRITICAL)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, std in zip(bars, phase3_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{std:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig_path = results_dir / "metrics_comparison.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {fig_path}")
    plt.close()
    
    # === SUMMARY TABLE ===
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY (Best from Pareto Front)")
    print(f"{'='*80}\n")
    
    for agent_name in agents:
        result = all_results[agent_name]
        metrics = result['metrics']
        phase3_mean = metrics['phase3_mean']
        phase3_std = metrics['phase3_std']
        trial_num = result['trial_number']
        pareto_size = result.get('pareto_front_size', '?')
        
        print(f"{agent_name:25} | Mean: {phase3_mean:7.2f} Mbps | Std: {phase3_std:6.2f} Mbps | Trial: {trial_num:3d} | Pareto: {pareto_size}")
    
    print(f"\n{'='*80}\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
