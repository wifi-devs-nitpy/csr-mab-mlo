"""
Optuna-based hyperparameter tuning for 10k simulation runs on scenario5.
Focus: Maximize avg throughput + Minimize variance (especially after 5k runs).
Multi-objective optimization for all 4 algorithms (UCB, EGreedy, Softmax, Thompson Sampling).
"""
import os
os.environ['JAX_ENABLE_X64'] = 'True'

from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
import json
from pathlib import Path
import optuna
from optuna.trial import Trial
from functools import partial

# ============================================================================
# SCENARIO SETUP
# ============================================================================
n_tx_power_levels = 12
scenario = simple_scenario_5(d_ap=30, d_sta=2, mcs=11, n_tx_power_levels=n_tx_power_levels)
total_steps = 10_000

# Global scenario reference for objective function
GLOBAL_SCENARIO = scenario

# ============================================================================
# SIMULATION FUNCTION
# ============================================================================
def run_simulation(agent_type, params_dict, seed=42):
    """
    Run a single simulation with given agent type and hyperparameters.
    Returns: throughput array (length total_steps + 1)
    """
    agent_factory = MapcAgentFactory(
        associations=GLOBAL_SCENARIO.associations,
        agent_type=agent_type,
        agent_params_lvl1=params_dict['params_lvl1'],
        agent_params_lvl2=params_dict['params_lvl2'],
        agent_params_lvl3=params_dict['params_lvl3'],
        agent_params_lvl4=params_dict['params_lvl4'],
        tx_power_levels=n_tx_power_levels
    )

    agent = agent_factory.create_hierarchical_mapc_agent()

    throughput = [200]  # Initialize with 200 Mbps
    key = jax.random.PRNGKey(seed=seed)

    for i in range(total_steps):
        key, run_key = jax.random.split(key)
        link_ap_sta = agent.sample(throughput[-1])
        data_rate = GLOBAL_SCENARIO(run_key, link_ap_sta)
        throughput.append(float(data_rate))

    return np.array(throughput)

# ============================================================================
# METRICS CALCULATION
# ============================================================================
def calculate_metrics(throughput_array):
    """
    Calculate comprehensive metrics for the simulation.
    throughput_array: numpy array of length 10001 (steps 0-10000)
    """
    metrics = {
        'total_steps': len(throughput_array) - 1,
    }

    # Overall metrics
    metrics['overall_mean'] = float(np.mean(throughput_array[1:]))
    metrics['overall_std'] = float(np.std(throughput_array[1:]))
    metrics['overall_var'] = float(np.var(throughput_array[1:]))

    # Phase 1: First 1000 runs (steps 1-1000)
    phase1 = throughput_array[1:1001]
    metrics['phase1_mean'] = float(np.mean(phase1))
    metrics['phase1_std'] = float(np.std(phase1))
    metrics['phase1_var'] = float(np.var(phase1))

    # Phase 2: 1k-5k runs (steps 1001-5000)
    phase2 = throughput_array[1001:5001]
    metrics['phase2_mean'] = float(np.mean(phase2))
    metrics['phase2_std'] = float(np.std(phase2))
    metrics['phase2_var'] = float(np.var(phase2))

    # Phase 3: 5k-10k runs (steps 5001-10000) - CRITICAL PHASE
    phase3 = throughput_array[5001:10001]
    metrics['phase3_mean'] = float(np.mean(phase3))
    metrics['phase3_std'] = float(np.std(phase3))
    metrics['phase3_var'] = float(np.var(phase3))

    # Improvement metrics (comparing phase 3 to phase 1)
    metrics['phase3_to_phase1_improvement'] = float(metrics['phase3_mean'] - metrics['phase1_mean'])

    return metrics

# ============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# ============================================================================
def objective_ucb(trial: Trial, seed: int) -> float:
    """Objective function for UCB hyperparameter tuning"""
    # Level 1 parameters
    c_l1 = trial.suggest_float('c_l1', 0.1, 5.0)
    gamma_l1 = trial.suggest_float('gamma_l1', 0.0, 1.0)

    # Level 2 parameters
    c_l2 = trial.suggest_float('c_l2', 0.1, 4.0)
    gamma_l2 = trial.suggest_float('gamma_l2', 0.0, 1.0)

    # Level 3 parameters
    c_l3 = trial.suggest_float('c_l3', 0.1, 3.0)
    gamma_l3 = trial.suggest_float('gamma_l3', 0.0, 1.0)

    # Level 4 parameters
    c_l4 = trial.suggest_float('c_l4', 0.01, 2.0)
    gamma_l4 = trial.suggest_float('gamma_l4', 0.0, 1.0)

    params = {
        'params_lvl1': {'c': c_l1, 'gamma': gamma_l1},
        'params_lvl2': {'c': c_l2, 'gamma': gamma_l2},
        'params_lvl3': {'c': c_l3, 'gamma': gamma_l3},
        'params_lvl4': {'c': c_l4, 'gamma': gamma_l4},
    }

    try:
        throughput = run_simulation(UCB, params, seed=seed)
        metrics = calculate_metrics(throughput)
        
        # Composite objective: maximize phase3 mean, minimize phase3 std
        # CRITICAL: Very strong weight on phase3 variance reduction
        objective_value = (
            metrics['phase3_mean'] * 0.35 +                    # Higher throughput in phase 3
            (1.0 / (metrics['phase3_std'] + 0.1)) * 25 +       # Very low variance (CRITICAL)
            metrics['phase3_to_phase1_improvement'] * 0.15      # Improvement bonus
        )
        return objective_value
    except Exception as e:
        print(f"Error in trial: {e}")
        return -np.inf

def objective_egreedy(trial: Trial, seed: int) -> float:
    """Objective function for EGreedy hyperparameter tuning"""
    # Level 1 parameters
    e_l1 = trial.suggest_float('e_l1', 0.001, 0.2, log=True)
    opt_start_l1 = trial.suggest_float('opt_start_l1', 5.0, 200.0)
    alpha_l1 = trial.suggest_float('alpha_l1', 0.5, 1.0)

    # Level 2 parameters
    e_l2 = trial.suggest_float('e_l2', 0.001, 0.15, log=True)
    opt_start_l2 = trial.suggest_float('opt_start_l2', 1.0, 120.0)
    alpha_l2 = trial.suggest_float('alpha_l2', 0.5, 1.0)

    # Level 3 parameters
    e_l3 = trial.suggest_float('e_l3', 0.0001, 0.1, log=True)
    opt_start_l3 = trial.suggest_float('opt_start_l3', 1.0, 80.0)
    alpha_l3 = trial.suggest_float('alpha_l3', 0.5, 1.0)

    # Level 4 parameters
    e_l4 = trial.suggest_float('e_l4', 0.0001, 0.05, log=True)
    opt_start_l4 = trial.suggest_float('opt_start_l4', 1.0, 40.0)
    alpha_l4 = trial.suggest_float('alpha_l4', 0.5, 1.0)

    params = {
        'params_lvl1': {'e': e_l1, 'optimistic_start': opt_start_l1, 'alpha': alpha_l1},
        'params_lvl2': {'e': e_l2, 'optimistic_start': opt_start_l2, 'alpha': alpha_l2},
        'params_lvl3': {'e': e_l3, 'optimistic_start': opt_start_l3, 'alpha': alpha_l3},
        'params_lvl4': {'e': e_l4, 'optimistic_start': opt_start_l4, 'alpha': alpha_l4},
    }

    try:
        throughput = run_simulation(EGreedy, params, seed=seed)
        metrics = calculate_metrics(throughput)
        
        objective_value = (
            metrics['phase3_mean'] * 0.35 +
            (1.0 / (metrics['phase3_std'] + 0.1)) * 25 +
            metrics['phase3_to_phase1_improvement'] * 0.15
        )
        return objective_value
    except Exception as e:
        print(f"Error in trial: {e}")
        return -np.inf

def objective_softmax(trial: Trial, seed: int) -> float:
    """Objective function for Softmax hyperparameter tuning"""
    # Level 1 parameters
    lr_l1 = trial.suggest_float('lr_l1', 0.01, 3.0, log=True)
    tau_l1 = trial.suggest_float('tau_l1', 0.1, 10.0)
    alpha_l1 = trial.suggest_float('alpha_l1', 0.5, 1.0)

    # Level 2 parameters
    lr_l2 = trial.suggest_float('lr_l2', 0.01, 2.0, log=True)
    tau_l2 = trial.suggest_float('tau_l2', 0.1, 8.0)
    alpha_l2 = trial.suggest_float('alpha_l2', 0.5, 1.0)

    # Level 3 parameters
    lr_l3 = trial.suggest_float('lr_l3', 0.01, 1.0, log=True)
    tau_l3 = trial.suggest_float('tau_l3', 0.1, 5.0)
    alpha_l3 = trial.suggest_float('alpha_l3', 0.5, 1.0)

    # Level 4 parameters
    lr_l4 = trial.suggest_float('lr_l4', 0.01, 0.5, log=True)
    tau_l4 = trial.suggest_float('tau_l4', 0.1, 3.0)
    alpha_l4 = trial.suggest_float('alpha_l4', 0.5, 1.0)

    params = {
        'params_lvl1': {'lr': lr_l1, 'tau': tau_l1, 'alpha': alpha_l1},
        'params_lvl2': {'lr': lr_l2, 'tau': tau_l2, 'alpha': alpha_l2},
        'params_lvl3': {'lr': lr_l3, 'tau': tau_l3, 'alpha': alpha_l3},
        'params_lvl4': {'lr': lr_l4, 'tau': tau_l4, 'alpha': alpha_l4},
    }

    try:
        throughput = run_simulation(Softmax, params, seed=seed)
        metrics = calculate_metrics(throughput)
        
        objective_value = (
            metrics['phase3_mean'] * 0.35 +
            (1.0 / (metrics['phase3_std'] + 0.1)) * 25 +
            metrics['phase3_to_phase1_improvement'] * 0.15
        )
        return objective_value
    except Exception as e:
        print(f"Error in trial: {e}")
        return -np.inf

def objective_thompson(trial: Trial, seed: int) -> float:
    """Objective function for Thompson Sampling hyperparameter tuning"""
    # Level 1 parameters
    alpha_l1 = trial.suggest_float('alpha_l1', 0.1, 10.0)
    beta_l1 = trial.suggest_float('beta_l1', 0.1, 10.0)
    mu_l1 = trial.suggest_float('mu_l1', 0.1, 5.0)

    # Level 2 parameters
    alpha_l2 = trial.suggest_float('alpha_l2', 0.1, 8.0)
    beta_l2 = trial.suggest_float('beta_l2', 0.1, 8.0)
    mu_l2 = trial.suggest_float('mu_l2', 0.1, 4.0)

    # Level 3 parameters
    alpha_l3 = trial.suggest_float('alpha_l3', 0.1, 6.0)
    beta_l3 = trial.suggest_float('beta_l3', 0.1, 6.0)
    mu_l3 = trial.suggest_float('mu_l3', 0.1, 3.0)

    # Level 4 parameters
    alpha_l4 = trial.suggest_float('alpha_l4', 0.1, 4.0)
    beta_l4 = trial.suggest_float('beta_l4', 0.1, 4.0)
    mu_l4 = trial.suggest_float('mu_l4', 0.1, 2.0)

    params = {
        'params_lvl1': {'alpha': alpha_l1, 'beta': beta_l1, 'lam': 1.0, 'mu': mu_l1},
        'params_lvl2': {'alpha': alpha_l2, 'beta': beta_l2, 'lam': 1.0, 'mu': mu_l2},
        'params_lvl3': {'alpha': alpha_l3, 'beta': beta_l3, 'lam': 1.0, 'mu': mu_l3},
        'params_lvl4': {'alpha': alpha_l4, 'beta': beta_l4, 'lam': 1.0, 'mu': mu_l4},
    }

    try:
        throughput = run_simulation(ThompsonSampling, params, seed=seed)
        metrics = calculate_metrics(throughput)
        
        objective_value = (
            metrics['phase3_mean'] * 0.35 +
            (1.0 / (metrics['phase3_std'] + 0.1)) * 25 +
            metrics['phase3_to_phase1_improvement'] * 0.15
        )
        return objective_value
    except Exception as e:
        print(f"Error in trial: {e}")
        return -np.inf

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    results_dir = Path('arrays/optimized_10k_optuna')
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OPTUNA-BASED HYPERPARAMETER TUNING FOR 10K SIMULATIONS")
    print("=" * 80)

    algorithms = {
        'UCB': (UCB, objective_ucb),
        'EGreedy': (EGreedy, objective_egreedy),
        'Softmax': (Softmax, objective_softmax),
        'NormalThompsonSampling': (ThompsonSampling, objective_thompson),
    }

    all_results = {}
    best_per_algorithm = {}
    best_throughput_arrays = {}

    for algo_name, (agent_type, objective_func) in algorithms.items():
        print(f"\n{'='*80}")
        print(f"TUNING: {algo_name}")
        print(f"{'='*80}")

        # Create Optuna study
        db_path = results_dir / f"optuna_{algo_name.lower()}.db"
        study = optuna.create_study(
            storage=f'sqlite:///{str(db_path)}',
            study_name=algo_name,
            load_if_exists=True,
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Run optimization
        objective_with_seed = partial(objective_func, seed=42)
        study.optimize(
            objective_with_seed,
            n_trials=50,  # 50 trials per algorithm
            show_progress_bar=True,
        )

        # Get best trial
        best_trial = study.best_trial
        best_params = best_trial.params
        print(f"\n  Best trial value: {best_trial.value:.4f}")
        print(f"  Best parameters:")
        for key, value in best_params.items():
            print(f"    {key}: {value}")

        # Extract best hyperparameters
        if algo_name == 'UCB':
            best_hp = {
                'name': f'{algo_name}_best',
                'params_lvl1': {'c': best_params['c_l1'], 'gamma': best_params['gamma_l1']},
                'params_lvl2': {'c': best_params['c_l2'], 'gamma': best_params['gamma_l2']},
                'params_lvl3': {'c': best_params['c_l3'], 'gamma': best_params['gamma_l3']},
                'params_lvl4': {'c': best_params['c_l4'], 'gamma': best_params['gamma_l4']},
            }
        elif algo_name == 'EGreedy':
            best_hp = {
                'name': f'{algo_name}_best',
                'params_lvl1': {'e': best_params['e_l1'], 'optimistic_start': best_params['opt_start_l1'], 'alpha': best_params['alpha_l1']},
                'params_lvl2': {'e': best_params['e_l2'], 'optimistic_start': best_params['opt_start_l2'], 'alpha': best_params['alpha_l2']},
                'params_lvl3': {'e': best_params['e_l3'], 'optimistic_start': best_params['opt_start_l3'], 'alpha': best_params['alpha_l3']},
                'params_lvl4': {'e': best_params['e_l4'], 'optimistic_start': best_params['opt_start_l4'], 'alpha': best_params['alpha_l4']},
            }
        elif algo_name == 'Softmax':
            best_hp = {
                'name': f'{algo_name}_best',
                'params_lvl1': {'lr': best_params['lr_l1'], 'tau': best_params['tau_l1'], 'alpha': best_params['alpha_l1']},
                'params_lvl2': {'lr': best_params['lr_l2'], 'tau': best_params['tau_l2'], 'alpha': best_params['alpha_l2']},
                'params_lvl3': {'lr': best_params['lr_l3'], 'tau': best_params['tau_l3'], 'alpha': best_params['alpha_l3']},
                'params_lvl4': {'lr': best_params['lr_l4'], 'tau': best_params['tau_l4'], 'alpha': best_params['alpha_l4']},
            }
        else:  # Thompson
            best_hp = {
                'name': f'{algo_name}_best',
                'params_lvl1': {'alpha': best_params['alpha_l1'], 'beta': best_params['beta_l1'], 'lam': 1.0, 'mu': best_params['mu_l1']},
                'params_lvl2': {'alpha': best_params['alpha_l2'], 'beta': best_params['beta_l2'], 'lam': 1.0, 'mu': best_params['mu_l2']},
                'params_lvl3': {'alpha': best_params['alpha_l3'], 'beta': best_params['beta_l3'], 'lam': 1.0, 'mu': best_params['mu_l3']},
                'params_lvl4': {'alpha': best_params['alpha_l4'], 'beta': best_params['beta_l4'], 'lam': 1.0, 'mu': best_params['mu_l4']},
            }

        # Run final simulation with best hyperparameters
        print(f"\n  Running final simulation with best hyperparameters...")
        best_throughput = run_simulation(agent_type, best_hp, seed=42)
        best_metrics = calculate_metrics(best_throughput)

        best_per_algorithm[algo_name] = {
            'hyperparams': best_hp,
            'metrics': best_metrics,
            'optuna_value': best_trial.value,
        }
        best_throughput_arrays[algo_name] = best_throughput

        # Save array
        array_path = results_dir / f"{algo_name}_best_10k.npy"
        jnp.save(str(array_path), best_throughput)

        # Print metrics
        print(f"\n  METRICS FOR BEST CONFIGURATION:")
        print(f"    Overall Mean: {best_metrics['overall_mean']:.2f} Mbps")
        print(f"    Overall Std:  {best_metrics['overall_std']:.2f} Mbps")
        print(f"    Phase 1 Mean: {best_metrics['phase1_mean']:.2f} Mbps")
        print(f"    Phase 3 (5k-10k) Mean: {best_metrics['phase3_mean']:.2f} Mbps")
        print(f"    Phase 3 (5k-10k) Std:  {best_metrics['phase3_std']:.2f} Mbps (CRITICAL)")
        print(f"    Phase 1->3 Improvement: {best_metrics['phase3_to_phase1_improvement']:.2f} Mbps")

    # ========================================================================
    # SAVE ALL RESULTS
    # ========================================================================
    print(f"\n{'='*80}")
    print("FINAL RESULTS - BEST HYPERPARAMETERS")
    print(f"{'='*80}\n")

    final_results = {}
    for algo_name, result_data in best_per_algorithm.items():
        final_results[algo_name] = {
            'hyperparameters': result_data['hyperparams'],
            'metrics': result_data['metrics'],
            'optuna_objective_value': result_data['optuna_value'],
        }
        print(f"\n{algo_name}:")
        print(f"  Optuna Score: {result_data['optuna_value']:.4f}")
        print(f"  Phase 3 Mean: {result_data['metrics']['phase3_mean']:.2f} Mbps")
        print(f"  Phase 3 Std:  {result_data['metrics']['phase3_std']:.2f} Mbps")
        print(f"  Improvement:  {result_data['metrics']['phase3_to_phase1_improvement']:.2f} Mbps")

    # Save to JSON
    results_file = results_dir / "best_hyperparameters_optuna.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n\nSaved results to: {results_file}")

    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*80}\n")

    # Plot 1: Best configurations comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Best Configurations Per Algorithm (10k steps) - Optuna Tuned', fontsize=16, fontweight='bold')

    for idx, algo_name in enumerate(['UCB', 'EGreedy', 'Softmax', 'NormalThompsonSampling']):
        ax = axes[idx // 2, idx % 2]
        best_metrics = best_per_algorithm[algo_name]['metrics']
        throughput = best_throughput_arrays[algo_name]

        # Plot with smoothing
        kernel = np.ones(100) / 100
        smoothed = np.convolve(throughput, kernel, mode='valid')
        ax.plot(np.arange(len(smoothed)) + 50, smoothed, linewidth=2.5, label=algo_name, color='blue')

        # Phase markers
        ax.axvline(x=1000, color='green', linestyle='--', alpha=0.6, linewidth=2, label='End Phase 1')
        ax.axvline(x=5000, color='red', linestyle='--', alpha=0.6, linewidth=2, label='CRITICAL: Start Phase 3')

        # Fill critical region
        ax.axvspan(5000, 10000, alpha=0.1, color='red')

        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel('Throughput (Mbps)', fontsize=11)
        ax.set_title(f"{algo_name}\nMean(Ph3): {best_metrics['phase3_mean']:.2f} Mbps | Std(Ph3): {best_metrics['phase3_std']:.2f} Mbps",
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = results_dir / "best_algorithms_comparison.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path.name}")
    plt.close()

    # Plot 2: Variance evolution (critical metric)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Variance Evolution Over Time - Best Configurations (CRITICAL METRIC)', fontsize=16, fontweight='bold')

    for idx, algo_name in enumerate(['UCB', 'EGreedy', 'Softmax', 'NormalThompsonSampling']):
        ax = axes[idx // 2, idx % 2]
        best_metrics = best_per_algorithm[algo_name]['metrics']
        throughput = best_throughput_arrays[algo_name]

        # Calculate rolling variance
        window = 100
        rolling_var = np.array([np.var(throughput[max(0, i-window):i]) for i in range(window, len(throughput))])
        rolling_std = np.sqrt(rolling_var)

        ax.plot(np.arange(len(rolling_std)) + window, rolling_std, linewidth=2, label='Rolling Std (w=100)', color='navy')
        ax.axvline(x=5000, color='red', linestyle='--', alpha=0.6, linewidth=2.5, label='Start Phase 3 (CRITICAL)')
        ax.axhline(y=best_metrics['phase3_std'], color='green', linestyle=':', alpha=0.7, linewidth=2, label=f'Phase 3 Avg Std: {best_metrics["phase3_std"]:.2f}')
        ax.axvspan(5000, 10000, alpha=0.1, color='orange')

        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel('Standard Deviation', fontsize=11)
        ax.set_title(f"{algo_name} (Phase 3 Std: {best_metrics['phase3_std']:.2f} Mbps)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = results_dir / "variance_evolution_critical.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path.name}")
    plt.close()

    # Plot 3: Phase comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance Comparison Across Phases', fontsize=14, fontweight='bold')

    algos = ['UCB', 'EGreedy', 'Softmax', 'NormalThompsonSampling']
    phase1_means = [best_per_algorithm[a]['metrics']['phase1_mean'] for a in algos]
    phase3_means = [best_per_algorithm[a]['metrics']['phase3_mean'] for a in algos]
    phase3_stds = [best_per_algorithm[a]['metrics']['phase3_std'] for a in algos]

    # Subplot 1: Mean throughput per phase
    x = np.arange(len(algos))
    width = 0.35
    ax = axes[0]
    ax.bar(x - width/2, phase1_means, width, label='Phase 1 Mean', alpha=0.8, color='skyblue')
    ax.bar(x + width/2, phase3_means, width, label='Phase 3 Mean', alpha=0.8, color='navy')
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Mean Throughput Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Phase 3 variance
    ax = axes[1]
    colors = ['red' if std > 50 else 'orange' if std > 30 else 'green' for std in phase3_stds]
    bars = ax.bar(algos, phase3_stds, color=colors, alpha=0.8)
    ax.set_ylabel('Standard Deviation (Mbps)', fontsize=11)
    ax.set_title('Phase 3 Variance (Lower is Better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, std in zip(bars, phase3_stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    fig_path = results_dir / "phase_comparison.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path.name}")
    plt.close()

    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE!")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
