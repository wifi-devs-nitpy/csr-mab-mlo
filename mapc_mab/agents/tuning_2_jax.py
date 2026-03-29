import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

from argparse import ArgumentParser
from functools import partial
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optuna
from reinforced_lib.agents.mab import *

from mapc_mab.agents import MapcAgentFactory
from mapc_mab.envs.dynamic_scenarios import random_scenario

import jax 
import jax.numpy as jnp
from mapc_mab.envs.static_scenarios import Scenario


TRAINING_SCENARIOS = [
    (random_scenario(seed=1, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500*2), 500*2),
    (random_scenario(seed=2, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000*2), 1000*2),
    (random_scenario(seed=3, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=4, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000*2), 2000*2),
    (random_scenario(seed=5, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=6, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000*2), 3000*2),
]

SLOTS_AHEAD = 1


def run_single_episode_optimized(agent, scenario, n_steps, key):
    """
    Run a single episode with proper agent state and reward flow.
    Optimized with pre-generated keys and minimal overhead.
    """
    scenario.reset()
    
    # Pre-generate all random keys for this episode (JAX optimization)
    keys = jax.random.split(key, n_steps)
    
    # Accumulate rewards efficiently
    total_reward = 0.
    reward = 0.  # Initial reward is 0 for first step
    
    # Main episode loop - agent learns from previous rewards
    for step_key in keys:
        # Agent samples action based on its internal state and previous reward
        link_ap_sta = agent.sample(reward)
        
        # Environment evaluates the action and returns throughput
        data_rate = scenario(key=step_key, link_ap_sta=link_ap_sta)
        
        # Extract the reward value
        reward = data_rate.item()
        total_reward += reward
    
    # Return mean reward for this episode
    return total_reward / n_steps


def run_scenario(
        agent_factory: MapcAgentFactory,
        scenario: Scenario,
        n_reps: int,
        n_steps: int,
        seed: int
) -> float:
    """
    Run multiple independent episodes (repetitions).
    Each repetition gets a fresh agent that learns during its episode.
    """
    key = jax.random.PRNGKey(seed)
    
    # Generate independent seeds for each repetition
    rep_keys = jax.random.split(key, n_reps)
    
    total_mean_reward = 0.
    
    # Run each repetition with its own agent
    for rep_key in rep_keys:
        # Create a fresh agent for this episode
        agent = agent_factory.create_mapc_agent()
        
        # Run the episode (agent learns from rewards during the episode)
        mean_reward = run_single_episode_optimized(agent, scenario, n_steps, rep_key)
        total_mean_reward += mean_reward
    
    # Average across all repetitions
    return total_mean_reward / n_reps


def run_episode_worker(args):
    """
    Worker function for parallel episode execution.
    Used when parallelizing across repetitions.
    """
    agent_factory, scenario, n_steps, rep_seed = args
    
    key = jax.random.PRNGKey(rep_seed)
    agent = agent_factory.create_mapc_agent()
    
    return run_single_episode_optimized(agent, scenario, n_steps, key)


def run_scenario_parallel_reps(
        agent_factory: MapcAgentFactory,
        scenario: Scenario,
        n_reps: int,
        n_steps: int,
        seed: int,
        n_workers: int = 4
) -> float:
    """
    Run repetitions in parallel using ProcessPoolExecutor.
    Each repetition runs in a separate process with its own agent.
    
    Use this when:
    - n_reps > 1
    - You have multiple CPU cores available
    - Episode runtime is significant (>1 second)
    """
    if n_reps == 1:
        # No benefit from parallelization
        return run_scenario(agent_factory, scenario, n_reps, n_steps, seed)
    
    key = jax.random.PRNGKey(seed)
    rep_seeds = jax.random.randint(key, (n_reps,), 0, 2**31 - 1)
    
    # Prepare arguments for each worker
    worker_args = [
        (agent_factory, scenario, n_steps, int(rep_seed))
        for rep_seed in rep_seeds
    ]
    
    # Run episodes in parallel
    with ProcessPoolExecutor(max_workers=min(n_workers, n_reps)) as executor:
        mean_rewards = list(executor.map(run_episode_worker, worker_args))
    
    return np.mean(mean_rewards)


def run_scenario_jax_scan(
        agent_factory: MapcAgentFactory,
        scenario: Scenario,
        n_reps: int,
        n_steps: int,
        seed: int
) -> float:
    """
    Alternative: Use JAX scan for the inner loop if agent supports it.
    This requires that agent.sample() and scenario() are JAX-traceable.
    
    Falls back to regular version if not compatible.
    """
    try:
        key = jax.random.PRNGKey(seed)
        total_mean_reward = 0.
        
        for rep in range(n_reps):
            key, rep_key = jax.random.split(key)
            agent = agent_factory.create_mapc_agent()
            scenario.reset()
            
            # Pre-generate keys for scan
            step_keys = jax.random.split(rep_key, n_steps)
            
            def step_fn(reward, step_key):
                """Single step in the episode"""
                link_ap_sta = agent.sample(reward)
                data_rate = scenario(key=step_key, link_ap_sta=link_ap_sta)
                new_reward = data_rate.item()
                return new_reward, new_reward
            
            # Use scan for the inner loop
            _, rewards = jax.lax.scan(step_fn, 0., step_keys)
            total_mean_reward += jnp.mean(rewards)
        
        return float(total_mean_reward / n_reps)
    
    except Exception as e:
        # Fall back to regular version
        print(f"JAX scan failed ({e}), using regular version")
        return run_scenario(agent_factory, scenario, n_reps, n_steps, seed)


def objective(trial: optuna.Trial, agent: str, hierarchical: bool, seed: int, 
              use_scan: bool = False, parallel_reps: bool = False) -> float:
    if agent == 'EGreedy':
        def suggest_params(level):
            return {
                'e': trial.suggest_float(f'e_{level}', 0.01, 0.1, log=True),
                'optimistic_start': trial.suggest_float(f'optimistic_start_{level}', 0., 100.),
                'alpha': trial.suggest_float(f'alpha_{level}', 0., 1.)
            }

        agent_type = EGreedy
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)
            agent_params_lvl4 = suggest_params(4)

    elif agent == 'Softmax':
        def suggest_params(level):
            return {
                'lr': trial.suggest_float(f'lr_{level}', 0.01, 10., log=True),
                'tau': trial.suggest_float(f'tau_{level}', 0.1, 10., log=True),
                'alpha': trial.suggest_float(f'alpha_{level}', 0., 1.)
            }

        agent_type = Softmax
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)
            agent_params_lvl4 = suggest_params(4)

    elif agent == 'UCB':
        def suggest_params(level):
            return {
                'c': trial.suggest_float(f'c_{level}', 0., 5.),
                'gamma': trial.suggest_float(f'gamma_{level}', 0., 1.)
            }

        agent_type = UCB
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)
            agent_params_lvl4 = suggest_params(4)

    elif agent == 'NormalThompsonSampling':
        def suggest_params(level):
            return {
                'alpha': trial.suggest_float(f'alpha_{level}', 0., 10.),
                'beta': trial.suggest_float(f'beta_{level}', 0., 10.),
                'lam': 1.,
                'mu': trial.suggest_float(f'mu_{level}', 0., 5.)
            }

        agent_type = NormalThompsonSampling
        agent_params_lvl1 = suggest_params(1)
        if hierarchical:
            agent_params_lvl2 = suggest_params(2)
            agent_params_lvl3 = suggest_params(3)
            agent_params_lvl4 = suggest_params(4)
    else:
        raise ValueError(f'Unknown agent {agent}')

    runs = []

    for step, (scenario, n_steps) in enumerate(TRAINING_SCENARIOS):
        if hierarchical:
            agent_factory = MapcAgentFactory(
                scenario.associations, agent_type, 
                agent_params_lvl1, agent_params_lvl2, 
                agent_params_lvl3, agent_params_lvl4, 
                hierarchical=True, seed=seed
            )
        else:
            agent_factory = MapcAgentFactory(
                scenario.associations, agent_type, 
                agent_params_lvl1, hierarchical=False, seed=seed
            )

        # Choose execution mode
        if use_scan:
            results = run_scenario_jax_scan(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=seed)
        elif parallel_reps:
            results = run_scenario_parallel_reps(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=seed)
        else:
            results = run_scenario(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=seed)
        
        runs.append(results)

        trial.report(results, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(runs)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-a', '--agent', type=str, required=True)
    args.add_argument('-d', '--database', type=str, required=True)
    args.add_argument('-f', '--flat', action='store_true', default=False)
    args.add_argument('-n', '--n_trials', type=int, default=200)
    args.add_argument('-s', '--seed', type=int, default=42)
    args.add_argument('-j', '--n_jobs', type=int, default=-1, 
                      help='Number of parallel Optuna trials (-1 for all cores)')

    args = args.parse_args()

    print(f"JAX devices: {jax.devices()}")
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Optuna will use {args.n_jobs if args.n_jobs > 0 else mp.cpu_count()} parallel workers")

    study = optuna.create_study(
        storage=f'sqlite:///{args.database}',
        study_name=args.agent,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed), 
        pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource=7, reduction_factor=3)
    )

    study.optimize(
        partial(objective, agent=args.agent, hierarchical=not args.flat, seed=args.seed,
                use_scan=True, parallel_reps=args.parallel_reps),
        n_trials=args.n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )
    
    print("\n" + "="*50)
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")