import os
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

from argparse import ArgumentParser
from functools import partial

import numpy as np
import optuna
from reinforced_lib.agents.mab import *

from mapc_mab.agents import MapcAgentFactory
from mapc_mab.envs.dynamic_scenarios import random_scenario

import jax 
from mapc_mab.envs.static_scenarios import Scenario


TRAINING_SCENARIOS = [
    (random_scenario(seed=1, d_ap=20., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500*2), 500*2),
    (random_scenario(seed=2, d_ap=20., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000*2), 1000*2),
    (random_scenario(seed=3, d_ap=20., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=4, d_ap=20., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000*2), 2000*2),
    (random_scenario(seed=5, d_ap=20., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=6, d_ap=20., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000*2), 3000*2),
    (random_scenario(seed=1, d_ap=30., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500*2), 500*2),
    (random_scenario(seed=2, d_ap=30., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000*2), 1000*2),
    (random_scenario(seed=3, d_ap=30., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=4, d_ap=30., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000*2), 2000*2),
    (random_scenario(seed=5, d_ap=30., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=6, d_ap=30., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000*2), 3000*2),
    (random_scenario(seed=1, d_ap=50., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500*2), 500*2),
    (random_scenario(seed=2, d_ap=50., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000*2), 1000*2),
    (random_scenario(seed=3, d_ap=50., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=4, d_ap=50., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000*2), 2000*2),
    (random_scenario(seed=5, d_ap=50., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=6, d_ap=50., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000*2), 3000*2),
    (random_scenario(seed=1, d_ap=60., d_sta=8., n_ap=2, n_sta_per_ap=5, max_steps=500*2), 500*2),
    (random_scenario(seed=2, d_ap=60., d_sta=5., n_ap=3, n_sta_per_ap=3, max_steps=1000*2), 1000*2),
    (random_scenario(seed=3, d_ap=60., d_sta=5., n_ap=3, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=4, d_ap=60., d_sta=5., n_ap=4, n_sta_per_ap=3, max_steps=2000*2), 2000*2),
    (random_scenario(seed=5, d_ap=60., d_sta=4., n_ap=4, n_sta_per_ap=4, max_steps=500*2), 500*2),
    (random_scenario(seed=6, d_ap=60., d_sta=4., n_ap=5, n_sta_per_ap=3, max_steps=3000*2), 3000*2),
]

SLOTS_AHEAD = 1


def run_scenario(
        agent_factory: MapcAgentFactory,
        scenario: Scenario,
        n_reps: int,
        n_steps: int,
        seed: int
) -> tuple[list, list]:
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    # for _ in range(n_reps):
    #     agent = agent_factory.create_mapc_agent()
    #     scenario.reset()
    #     runs.append([])
    #     actions.append([])
    #     reward = 0.

    #     for _ in range(n_steps):
    #         key, scenario_key = jax.random.split(key)
    #         link_ap_sta = agent.sample(reward)
    #         data_rate = scenario(key=scenario_key, link_ap_sta=link_ap_sta)
    #         runs[-1].append(data_rate.item())


    # return runs, actions

    agent = agent_factory.create_mapc_agent()
    scenario.reset()
    reward = 0.

    for _ in range(n_steps):
            key, scenario_key = jax.random.split(key)
            link_ap_sta = agent.sample(reward)
            data_rate = scenario(key=scenario_key, link_ap_sta=link_ap_sta)
            reward += data_rate.item()/n_steps
        
    return reward


def objective(trial: optuna.Trial, agent: str, hierarchical: bool, seed: int) -> float:
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
            agent_factory = MapcAgentFactory(scenario.associations, agent_type, agent_params_lvl1, agent_params_lvl2, agent_params_lvl3, agent_params_lvl4, hierarchical=True, seed=seed)
        else:
            agent_factory = MapcAgentFactory(scenario.associations, agent_type, agent_params_lvl1, hierarchical=False, seed=seed)

        # results = np.mean(run_scenario(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=seed)[0])
        results = np.mean(run_scenario(agent_factory, scenario, n_reps=1, n_steps=n_steps, seed=seed))

        runs.append(results)
        trial.report(results, step)

    return np.mean(runs)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-a', '--agent', type=str, required=True)
    args.add_argument('-d', '--database', type=str, required=True)
    args.add_argument('-f', '--flat', action='store_true', default=False)
    args.add_argument('-n', '--n_trials', type=int, default=200)
    args.add_argument('-s', '--seed', type=int, default=42)
    args = args.parse_args()

    study = optuna.create_study(
        storage=f'sqlite:///{args.database}',
        study_name=args.agent,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed), 
    )

    study.optimize(
        partial(objective, agent=args.agent, hierarchical=not args.flat, seed=args.seed),
        n_trials=args.n_trials,
        show_progress_bar=True
    )
