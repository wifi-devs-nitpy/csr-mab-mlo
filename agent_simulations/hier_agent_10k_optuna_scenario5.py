"""
Optuna-based hyperparameter tuning for 10k simulation runs on scenario5.
Optimizes for: Maximum avg throughput + Minimum variance (especially after 5k runs).
Based on tuning_2.py pattern.
"""
import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'

from argparse import ArgumentParser
from functools import partial

import numpy as np
import optuna
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling

from mapc_mab.agents import MapcAgentFactory
from mapc_mab.envs.static_scenarios import simple_scenario_5

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONSTANTS
# ============================================================================
N_TX_POWER_LEVELS = 12
TOTAL_STEPS = 10_000
REPORT_INTERVAL = 1000  # Report metrics every 1000 steps
N_TRIALS_PER_ALGORITHM = 100  # Increased from 50 to explore wider space


# ============================================================================
# SCENARIO SETUP
# ============================================================================
def create_scenario():
    return simple_scenario_5(d_ap=30, d_sta=2, mcs=11, n_tx_power_levels=N_TX_POWER_LEVELS)


# ============================================================================
# SIMULATION FUNCTION
# ============================================================================
def run_simulation_with_reporting(agent_type, params_dict, seed=42, trial=None):
    """
    Run simulation and report intermediate metrics every REPORT_INTERVAL steps.
    This allows Optuna to prune unpromising trials early.
    
    Returns: (throughput_array, phase3_metric)
    """
    scenario = create_scenario()
    
    agent_factory = MapcAgentFactory(
        associations=scenario.associations,
        agent_type=agent_type,
        agent_params_lvl1=params_dict['params_lvl1'],
        agent_params_lvl2=params_dict['params_lvl2'],
        agent_params_lvl3=params_dict['params_lvl3'],
        agent_params_lvl4=params_dict['params_lvl4'],
        tx_power_levels=N_TX_POWER_LEVELS
    )

    agent = agent_factory.create_hierarchical_mapc_agent()

    throughput = [200]  # Initialize with 200 Mbps
    key = jax.random.PRNGKey(seed=seed)
    
    phase3_start = 5000
    phase3_results = []

    for i in range(1, TOTAL_STEPS + 1):
        key, run_key = jax.random.split(key)
        link_ap_sta = agent.sample(throughput[-1])
        data_rate = scenario(run_key, link_ap_sta)
        throughput.append(float(data_rate))
        
        # Collect phase 3 results for early evaluation
        if i > phase3_start:
            phase3_results.append(float(data_rate))

        # Report intermediate result every REPORT_INTERVAL steps
        if trial is not None and i % REPORT_INTERVAL == 0:
            # Early phase 3 metric (as soon as we have data)
            if len(phase3_results) > 0:
                phase3_mean = np.mean(phase3_results)
                phase3_std = np.std(phase3_results) if len(phase3_results) > 1 else 0
                
                # Composite metric for early feedback
                metric_value = (
                    phase3_mean * 0.35 +
                    (1.0 / (phase3_std + 0.1)) * 25 +
                    (phase3_mean - 200) * 0.15  # Improvement from initial
                )
    return np.array(throughput)


# ============================================================================
# METRICS CALCULATION
# ============================================================================
def calculate_metrics(throughput_array):
    """Calculate comprehensive metrics from simulation results."""
    metrics = {}

    # Overall metrics
    metrics['overall_mean'] = float(np.mean(throughput_array[1:]))
    metrics['overall_std'] = float(np.std(throughput_array[1:]))
    
    # Phase 1: First 1000 runs
    phase1 = throughput_array[1:1001]
    metrics['phase1_mean'] = float(np.mean(phase1))
    metrics['phase1_std'] = float(np.std(phase1))

    # Phase 2: 1k-5k runs
    phase2 = throughput_array[1001:5001]
    metrics['phase2_mean'] = float(np.mean(phase2))
    metrics['phase2_std'] = float(np.std(phase2))

    # Phase 3: 5k-10k runs (CRITICAL)
    phase3 = throughput_array[5001:10001]
    metrics['phase3_mean'] = float(np.mean(phase3))
    metrics['phase3_std'] = float(np.std(phase3))
    
    # Improvement metric
    metrics['phase3_to_phase1_improvement'] = float(metrics['phase3_mean'] - metrics['phase1_mean'])

    return metrics


# ============================================================================
# OPTUNA OBJECTIVE FUNCTIONS
# ============================================================================
def objective_ucb(trial: optuna.Trial, seed: int) -> tuple:
    """UCB multi-objective optimization - EXPANDED RANGES
    Returns: (phase3_mean_to_maximize, phase3_std_to_minimize)
    """
    # Level 1 parameters - more aggressive exploration
    c_l1 = trial.suggest_float('c_l1', 0.01, 10.0)
    gamma_l1 = trial.suggest_float('gamma_l1', 0.0, 1.0)

    # Level 2 parameters
    c_l2 = trial.suggest_float('c_l2', 0.01, 8.0)
    gamma_l2 = trial.suggest_float('gamma_l2', 0.0, 1.0)

    # Level 3 parameters
    c_l3 = trial.suggest_float('c_l3', 0.01, 6.0)
    gamma_l3 = trial.suggest_float('gamma_l3', 0.0, 1.0)

    # Level 4 parameters
    c_l4 = trial.suggest_float('c_l4', 0.001, 4.0)
    gamma_l4 = trial.suggest_float('gamma_l4', 0.0, 1.0)

    params = {
        'params_lvl1': {'c': c_l1, 'gamma': gamma_l1},
        'params_lvl2': {'c': c_l2, 'gamma': gamma_l2},
        'params_lvl3': {'c': c_l3, 'gamma': gamma_l3},
        'params_lvl4': {'c': c_l4, 'gamma': gamma_l4},
    }

    throughput = run_simulation_with_reporting(UCB, params, seed=seed, trial=trial)
    metrics = calculate_metrics(throughput)
    
    # Return tuple: (maximize phase3 mean, minimize phase3 std)
    return [metrics['phase3_mean'], metrics['phase3_std']]


def objective_egreedy(trial: optuna.Trial, seed: int) -> tuple:
    """EGreedy multi-objective optimization - EXPANDED RANGES
    Returns: (phase3_mean_to_maximize, phase3_std_to_minimize)
    """
    # Level 1 parameters - much broader exploration
    e_l1 = trial.suggest_float('e_l1', 0.0001, 0.5, log=True)
    opt_start_l1 = trial.suggest_float('opt_start_l1', 1.0, 500.0)
    alpha_l1 = trial.suggest_float('alpha_l1', 0.0, 1.0)

    # Level 2 parameters
    e_l2 = trial.suggest_float('e_l2', 0.0001, 0.4, log=True)
    opt_start_l2 = trial.suggest_float('opt_start_l2', 1.0, 300.0)
    alpha_l2 = trial.suggest_float('alpha_l2', 0.0, 1.0)

    # Level 3 parameters
    e_l3 = trial.suggest_float('e_l3', 0.00001, 0.2, log=True)
    opt_start_l3 = trial.suggest_float('opt_start_l3', 1.0, 200.0)
    alpha_l3 = trial.suggest_float('alpha_l3', 0.0, 1.0)

    # Level 4 parameters
    e_l4 = trial.suggest_float('e_l4', 0.00001, 0.1, log=True)
    opt_start_l4 = trial.suggest_float('opt_start_l4', 1.0, 150.0)
    alpha_l4 = trial.suggest_float('alpha_l4', 0.0, 1.0)

    params = {
        'params_lvl1': {'e': e_l1, 'optimistic_start': opt_start_l1, 'alpha': alpha_l1},
        'params_lvl2': {'e': e_l2, 'optimistic_start': opt_start_l2, 'alpha': alpha_l2},
        'params_lvl3': {'e': e_l3, 'optimistic_start': opt_start_l3, 'alpha': alpha_l3},
        'params_lvl4': {'e': e_l4, 'optimistic_start': opt_start_l4, 'alpha': alpha_l4},
    }

    throughput = run_simulation_with_reporting(EGreedy, params, seed=seed, trial=trial)
    metrics = calculate_metrics(throughput)
    
    # Return tuple: (maximize phase3 mean, minimize phase3 std)
    return [metrics['phase3_mean'], metrics['phase3_std']]


def objective_softmax(trial: optuna.Trial, seed: int) -> tuple:
    """Softmax multi-objective optimization - EXPANDED RANGES
    Returns: (phase3_mean_to_maximize, phase3_std_to_minimize)
    """
    # Level 1 parameters - much broader exploration
    lr_l1 = trial.suggest_float('lr_l1', 0.001, 20.0, log=True)
    tau_l1 = trial.suggest_float('tau_l1', 0.01, 50.0)
    alpha_l1 = trial.suggest_float('alpha_l1', 0.0, 1.0)

    # Level 2 parameters
    lr_l2 = trial.suggest_float('lr_l2', 0.001, 15.0, log=True)
    tau_l2 = trial.suggest_float('tau_l2', 0.01, 40.0)
    alpha_l2 = trial.suggest_float('alpha_l2', 0.0, 1.0)

    # Level 3 parameters
    lr_l3 = trial.suggest_float('lr_l3', 0.001, 10.0, log=True)
    tau_l3 = trial.suggest_float('tau_l3', 0.01, 30.0)
    alpha_l3 = trial.suggest_float('alpha_l3', 0.0, 1.0)

    # Level 4 parameters
    lr_l4 = trial.suggest_float('lr_l4', 0.001, 5.0, log=True)
    tau_l4 = trial.suggest_float('tau_l4', 0.01, 20.0)
    alpha_l4 = trial.suggest_float('alpha_l4', 0.0, 1.0)

    params = {
        'params_lvl1': {'lr': lr_l1, 'tau': tau_l1, 'alpha': alpha_l1},
        'params_lvl2': {'lr': lr_l2, 'tau': tau_l2, 'alpha': alpha_l2},
        'params_lvl3': {'lr': lr_l3, 'tau': tau_l3, 'alpha': alpha_l3},
        'params_lvl4': {'lr': lr_l4, 'tau': tau_l4, 'alpha': alpha_l4},
    }

    throughput = run_simulation_with_reporting(Softmax, params, seed=seed, trial=trial)
    metrics = calculate_metrics(throughput)
    
    # Return tuple: (maximize phase3 mean, minimize phase3 std)
    return [metrics['phase3_mean'], metrics['phase3_std']]


def objective_thompson(trial: optuna.Trial, seed: int) -> tuple:
    """Thompson Sampling multi-objective optimization - EXPANDED RANGES
    Returns: (phase3_mean_to_maximize, phase3_std_to_minimize)
    """
    # Level 1 parameters - much broader exploration
    alpha_l1 = trial.suggest_float('alpha_l1', 0.01, 50.0)
    beta_l1 = trial.suggest_float('beta_l1', 0.01, 50.0)
    mu_l1 = trial.suggest_float('mu_l1', 0.01, 20.0)

    # Level 2 parameters
    alpha_l2 = trial.suggest_float('alpha_l2', 0.01, 40.0)
    beta_l2 = trial.suggest_float('beta_l2', 0.01, 40.0)
    mu_l2 = trial.suggest_float('mu_l2', 0.01, 15.0)

    # Level 3 parameters
    alpha_l3 = trial.suggest_float('alpha_l3', 0.01, 30.0)
    beta_l3 = trial.suggest_float('beta_l3', 0.01, 30.0)
    mu_l3 = trial.suggest_float('mu_l3', 0.01, 12.0)

    # Level 4 parameters
    alpha_l4 = trial.suggest_float('alpha_l4', 0.01, 20.0)
    beta_l4 = trial.suggest_float('beta_l4', 0.01, 20.0)
    mu_l4 = trial.suggest_float('mu_l4', 0.01, 10.0)

    params = {
        'params_lvl1': {'alpha': alpha_l1, 'beta': beta_l1, 'lam': 1.0, 'mu': mu_l1},
        'params_lvl2': {'alpha': alpha_l2, 'beta': beta_l2, 'lam': 1.0, 'mu': mu_l2},
        'params_lvl3': {'alpha': alpha_l3, 'beta': beta_l3, 'lam': 1.0, 'mu': mu_l3},
        'params_lvl4': {'alpha': alpha_l4, 'beta': beta_l4, 'lam': 1.0, 'mu': mu_l4},
    }

    throughput = run_simulation_with_reporting(ThompsonSampling, params, seed=seed, trial=trial)
    metrics = calculate_metrics(throughput)
    
    # Return tuple: (maximize phase3 mean, minimize phase3 std)
    return [metrics['phase3_mean'], metrics['phase3_std']]


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    args = ArgumentParser()
    args.add_argument('-a', '--agent', type=str, required=True, 
                     help='Agent type: UCB, EGreedy, Softmax, or NormalThompsonSampling')
    args.add_argument('-d', '--database', type=str, required=True,
                     help='Path to Optuna database file')
    args.add_argument('-n', '--n_trials', type=int, default=N_TRIALS_PER_ALGORITHM,
                     help='Number of trials per algorithm')
    args.add_argument('-s', '--seed', type=int, default=42,
                     help='Random seed')
    args = args.parse_args()

    # Create results directory
    results_dir = Path('arrays/optuna_10k_scenario5')
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"OPTUNA TUNING FOR SCENARIO5 - 10K RUNS")
    print(f"Agent: {args.agent}")
    print(f"Trials: {args.n_trials}")
    print("=" * 80)

    # Map agent names to types and objective functions
    agent_map = {
        'UCB': (UCB, objective_ucb),
        'EGreedy': (EGreedy, objective_egreedy),
        'Softmax': (Softmax, objective_softmax),
        'NormalThompsonSampling': (ThompsonSampling, objective_thompson),
    }

    if args.agent not in agent_map:
        raise ValueError(f"Unknown agent: {args.agent}. Choose from {list(agent_map.keys())}")

    agent_type, objective_func = agent_map[args.agent]
    algo_name = args.agent
    # Create Optuna study with MULTI-OBJECTIVE optimization
    db_path = results_dir / f"optuna_{algo_name.lower()}.db"
    study = optuna.create_study(
        storage=f'sqlite:///{str(db_path)}',
        study_name=algo_name,
        load_if_exists=True,
        directions=['maximize', 'minimize'],  # Maximize mean, minimize std
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.HyperbandPruner(min_resource=2, max_resource=10, reduction_factor=3)
    )

    # Run optimization
    objective_with_seed = partial(objective_func, seed=args.seed)
    study.optimize(
        objective_with_seed,
        n_trials=args.n_trials,
        n_jobs=1,
        show_progress_bar=True,
    )

    # Get Pareto front (all non-dominated solutions)
    trials_df = study.trials_dataframe()
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    
    # For Pareto front: find trial with best trade-off
    # Normalize both objectives to [0, 1] and find solution closest to ideal point (1, 0)
    best_trials_pareto = study.best_trials
    
    print(f"\n  Pareto Front Size: {len(best_trials_pareto)} trials")
    print(f"  Total Completed Trials: {len(completed_trials)}")
    
    if not best_trials_pareto:
        print(f"  ERROR: No trials in Pareto front!")
        return
    
    # Among Pareto front, select the one that balances both objectives best
    # Score = (normalized_mean + (1-normalized_std))
    best_trial = None
    best_balance_score = -float('inf')
    
    means = [t.values[0] for t in best_trials_pareto]
    stds = [t.values[1] for t in best_trials_pareto]
    
    mean_min, mean_max = min(means), max(means)
    std_min, std_max = min(stds), max(stds)
    
    for trial in best_trials_pareto:
        mean_val = trial.values[0]
        std_val = trial.values[1]
        
        # Normalize
        norm_mean = (mean_val - mean_min) / (mean_max - mean_min + 1e-6)
        norm_std = (std_val - std_min) / (std_max - std_min + 1e-6)
        
        # Balance score: prioritize low variance (minimize std)
        balance_score = norm_mean * 0.4 + (1.0 - norm_std) * 0.6
        
        if balance_score > best_balance_score:
            best_balance_score = balance_score
            best_trial = trial
    
    best_params = best_trial.params
    print(f"\n  Selected Trial #{best_trial.number} from Pareto Front")
    print(f"    Phase3 Mean: {best_trial.values[0]:.2f} Mbps")
    print(f"    Phase3 Std:  {best_trial.values[1]:.2f} Mbps")
    print(f"    Balance Score: {best_balance_score:.4f}")
    print(f"\n  Best parameters:")
    for key, value in sorted(best_params.items()):
        print(f"    {key}: {value}")

    # Extract and organize best hyperparameters
    if args.agent == 'UCB':
        best_hp = {
            'params_lvl1': {'c': best_params['c_l1'], 'gamma': best_params['gamma_l1']},
            'params_lvl2': {'c': best_params['c_l2'], 'gamma': best_params['gamma_l2']},
            'params_lvl3': {'c': best_params['c_l3'], 'gamma': best_params['gamma_l3']},
            'params_lvl4': {'c': best_params['c_l4'], 'gamma': best_params['gamma_l4']},
        }
    elif args.agent == 'EGreedy':
        best_hp = {
            'params_lvl1': {'e': best_params['e_l1'], 'optimistic_start': best_params['opt_start_l1'], 'alpha': best_params['alpha_l1']},
            'params_lvl2': {'e': best_params['e_l2'], 'optimistic_start': best_params['opt_start_l2'], 'alpha': best_params['alpha_l2']},
            'params_lvl3': {'e': best_params['e_l3'], 'optimistic_start': best_params['opt_start_l3'], 'alpha': best_params['alpha_l3']},
            'params_lvl4': {'e': best_params['e_l4'], 'optimistic_start': best_params['opt_start_l4'], 'alpha': best_params['alpha_l4']},
        }
    elif args.agent == 'Softmax':
        best_hp = {
            'params_lvl1': {'lr': best_params['lr_l1'], 'tau': best_params['tau_l1'], 'alpha': best_params['alpha_l1']},
            'params_lvl2': {'lr': best_params['lr_l2'], 'tau': best_params['tau_l2'], 'alpha': best_params['alpha_l2']},
            'params_lvl3': {'lr': best_params['lr_l3'], 'tau': best_params['tau_l3'], 'alpha': best_params['alpha_l3']},
            'params_lvl4': {'lr': best_params['lr_l4'], 'tau': best_params['tau_l4'], 'alpha': best_params['alpha_l4']},
        }
    else:  # Thompson
        best_hp = {
            'params_lvl1': {'alpha': best_params['alpha_l1'], 'beta': best_params['beta_l1'], 'lam': 1.0, 'mu': best_params['mu_l1']},
            'params_lvl2': {'alpha': best_params['alpha_l2'], 'beta': best_params['beta_l2'], 'lam': 1.0, 'mu': best_params['mu_l2']},
            'params_lvl3': {'alpha': best_params['alpha_l3'], 'beta': best_params['beta_l3'], 'lam': 1.0, 'mu': best_params['mu_l3']},
            'params_lvl4': {'alpha': best_params['alpha_l4'], 'beta': best_params['beta_l4'], 'lam': 1.0, 'mu': best_params['mu_l4']},
        }

    # Run final simulation with best hyperparameters
    print(f"\n{'='*80}")
    print(f"RUNNING FINAL SIMULATION WITH BEST HYPERPARAMETERS")
    print(f"{'='*80}")
    
    throughput = run_simulation_with_reporting(agent_type, best_hp, seed=args.seed, trial=None)
    metrics = calculate_metrics(throughput)

    print(f"\nFinal Metrics:")
    print(f"  Overall Mean:           {metrics['overall_mean']:.2f} Mbps")
    print(f"  Overall Std:            {metrics['overall_std']:.2f} Mbps")
    print(f"  Phase 1 Mean:           {metrics['phase1_mean']:.2f} Mbps")
    print(f"  Phase 3 Mean (5k-10k):  {metrics['phase3_mean']:.2f} Mbps ⭐")
    print(f"  Phase 3 Std (5k-10k):   {metrics['phase3_std']:.2f} Mbps ⭐⭐⭐ (CRITICAL)")
    print(f"  Phase 1→3 Improvement:  {metrics['phase3_to_phase1_improvement']:+.2f} Mbps")

    # Save results
    import json
    results = {
        'agent': args.agent,
        'trial_number': best_trial.number,
        'pareto_front_size': len(best_trials_pareto),
        'objectives': {
            'phase3_mean_mbps': best_trial.values[0],
            'phase3_std_mbps': best_trial.values[1],
        },
        'balance_score': best_balance_score,
        'hyperparameters': best_hp,
        'metrics': metrics,
    }
    
    results_file = results_dir / f"best_{args.agent.lower()}_params.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save Pareto front info
    pareto_file = results_dir / f"pareto_front_{args.agent.lower()}.json"
    pareto_data = {
        'algorithm': args.agent,
        'n_trials': len(best_trials_pareto),
        'trials': [
            {
                'trial_id': t.number,
                'phase3_mean': t.values[0],
                'phase3_std': t.values[1],
            }
            for t in best_trials_pareto
        ]
    }
    with open(pareto_file, 'w') as f:
        json.dump(pareto_data, f, indent=2)
    
    # Save throughput array
    array_file = results_dir / f"{args.agent}_best_throughput.npy"
    jnp.save(str(array_file), throughput)

    print(f"\nResults saved to: {results_file}")
    print(f"Array saved to:  {array_file}")

    # Create visualization with Pareto front
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{args.agent} - Best Configuration + Pareto Front Analysis', fontsize=14, fontweight='bold')

    # Subplot 1: Throughput over time
    ax = axes[0, 0]
    kernel = np.ones(100) / 100
    smoothed = np.convolve(throughput, kernel, mode='valid')
    ax.plot(np.arange(len(smoothed)) + 50, smoothed, linewidth=2, color='navy', label='Smoothed (w=100)')
    ax.axvline(x=1000, color='green', linestyle='--', linewidth=2, alpha=0.7, label='End Phase 1')
    ax.axvline(x=5000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Start Phase 3 (CRITICAL)')
    ax.axvspan(5000, 10000, alpha=0.1, color='red')
    ax.set_xlabel('Steps', fontsize=11)
    ax.set_ylabel('Throughput (Mbps)', fontsize=11)
    ax.set_title('Best Trial: Throughput Evolution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Subplot 2: Variance over time
    ax = axes[0, 1]
    window = 100
    rolling_var = np.array([np.var(throughput[max(0, i-window):i]) for i in range(window, len(throughput))])
    rolling_std = np.sqrt(rolling_var)
    ax.plot(np.arange(len(rolling_std)) + window, rolling_std, linewidth=2, color='darkred', label='Rolling Std')
    ax.axvline(x=5000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Start Phase 3')
    ax.axhline(y=metrics['phase3_std'], color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Phase 3 Std: {metrics["phase3_std"]:.2f}')
    ax.axvspan(5000, 10000, alpha=0.1, color='red')
    ax.set_xlabel('Steps', fontsize=11)
    ax.set_ylabel('Standard Deviation', fontsize=11)
    ax.set_title('Best Trial: Variance Evolution (CRITICAL)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Pareto Front visualization
    ax = axes[1, 0]
    pareto_means = [t.values[0] for t in best_trials_pareto]
    pareto_stds = [t.values[1] for t in best_trials_pareto]
    
    # Plot all Pareto front trials
    ax.scatter(pareto_stds, pareto_means, s=100, alpha=0.6, c='lightblue', edgecolor='navy', linewidth=2, label='Pareto Front Trials')
    
    # Highlight selected trial
    ax.scatter([best_trial.values[1]], [best_trial.values[0]], s=300, c='red', marker='*', 
              edgecolor='darkred', linewidth=2, label=f'Selected Trial #{best_trial.number}', zorder=5)
    
    ax.set_xlabel('Phase 3 Std (Mbps) - Lower is Better →', fontsize=11)
    ax.set_ylabel('Phase 3 Mean (Mbps) - Higher is Better ↑', fontsize=11)
    ax.set_title(f'Pareto Front ({len(best_trials_pareto)} trials)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    # Subplot 4: Metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
BEST CONFIGURATION METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trial #: {best_trial.number} (from Pareto front)
Balance Score: {best_balance_score:.4f}

OBJECTIVES:
  Phase 3 Mean: {metrics['phase3_mean']:.2f} Mbps ↑
  Phase 3 Std:  {metrics['phase3_std']:.2f} Mbps ↓

OVERALL PERFORMANCE:
  Overall Mean: {metrics['overall_mean']:.2f} Mbps
  Overall Std:  {metrics['overall_std']:.2f} Mbps
  
PHASE IMPROVEMENTS:
  Phase 1 Mean: {metrics['phase1_mean']:.2f} Mbps
  Phase 3 Mean: {metrics['phase3_mean']:.2f} Mbps
  Improvement: {metrics['phase3_to_phase1_improvement']:+.2f} Mbps

PARETO FRONT:
  Total Size: {len(best_trials_pareto)} trials
  Mean Range: [{min(pareto_means):.1f}, {max(pareto_means):.1f}]
  Std Range:  [{min(pareto_stds):.1f}, {max(pareto_stds):.1f}]
    """
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig_file = results_dir / f"{args.agent}_best_trial.png"
    plt.savefig(str(fig_file), dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {fig_file}")
    plt.close()

    print(f"\n{'='*80}")
    print("TUNING COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
