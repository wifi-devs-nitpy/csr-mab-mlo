import optuna
import jax
import jax.numpy as jnp
import numpy as np
from mapc_mab.envs.static_scenarios import simple_scenario_5
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import UCB

# Configuration
TOTAL_STEPS = 5000
BURN_IN = 2000
BLOCK_SIZE = 100
N_TX_POWER_LEVELS = 12
SEED = 42

# Objective weights
LAMBDA_STD = 0.5
LAMBDA_VOL = 0.2
LAMBDA_DRIFT = 0.2
LAMBDA_REGRET = 0.1


def run_simulation(agent_factory):
    scenario = simple_scenario_5(
        d_ap=30,
        d_sta=2,
        mcs=11,
        n_tx_power_levels=N_TX_POWER_LEVELS
    )

    agent = agent_factory.create_hierarchical_mapc_agent()
    throughput = [0.0]
    key = jax.random.PRNGKey(SEED)

    for _ in range(1, TOTAL_STEPS):
        key, subkey = jax.random.split(key)
        action = agent.sample(throughput[-1])
        reward = scenario(subkey, action)
        throughput.append(reward)

    return jnp.asarray(throughput)


def compute_metrics(throughput):
    stable = throughput[BURN_IN:]

    # Mean and standard deviation
    mean_tp = jnp.mean(stable)
    std_tp = jnp.std(stable)

    # Temporal volatility (smoothness)
    volatility = jnp.mean(jnp.abs(jnp.diff(stable)))

    # Block drift
    num_blocks = len(stable) // BLOCK_SIZE
    trimmed = stable[:num_blocks * BLOCK_SIZE]
    blocks = trimmed.reshape(num_blocks, BLOCK_SIZE)
    block_means = jnp.mean(blocks, axis=1)
    drift = jnp.std(block_means)

    # Regret proxy
    regret = jnp.max(stable) - mean_tp

    return (
        float(mean_tp),
        float(std_tp),
        float(volatility),
        float(drift),
        float(regret),
    )


def objective(trial):
    scenario = simple_scenario_5(
        d_ap=30,
        d_sta=2,
        mcs=None,
        n_tx_power_levels=N_TX_POWER_LEVELS
    )

    agent_factory = MapcAgentFactory(
        associations=scenario.associations,
        agent_type=UCB,
        agent_params_lvl1={
            "c": trial.suggest_float("c_lvl1", 0.01, 5.0),
            "gamma": trial.suggest_float("gamma_lvl1", 0.01, 1.0),
        },
        agent_params_lvl2={
            "c": trial.suggest_float("c_lvl2", 0.01, 5.0),
            "gamma": trial.suggest_float("gamma_lvl2", 0.01, 1.0),
        },
        agent_params_lvl3={
            "c": trial.suggest_float("c_lvl3", 0.01, 5.0),
            "gamma": trial.suggest_float("gamma_lvl3", 0.01, 1.0),
        },
        agent_params_lvl4={
            "c": trial.suggest_float("c_lvl4", 0.01, 5.0),
            "gamma": trial.suggest_float("gamma_lvl4", 0.01, 1.0),
        },
        tx_power_levels=N_TX_POWER_LEVELS,
    )

    throughput = run_simulation(agent_factory)

    mean_tp, std_tp, vol, drift, regret = compute_metrics(throughput)

    # Composite objective
    score = (
        mean_tp
        - LAMBDA_STD * std_tp
        - LAMBDA_VOL * vol
        - LAMBDA_DRIFT * drift
        - LAMBDA_REGRET * regret
    )

    # Store diagnostics
    trial.set_user_attr("mean", mean_tp)
    trial.set_user_attr("std", std_tp)
    trial.set_user_attr("volatility", vol)
    trial.set_user_attr("drift", drift)
    trial.set_user_attr("regret", regret)
    trial.set_user_attr("score", score)

    return score


study = optuna.create_study(
    storage=f'sqlite:///ucb_s5_fine_tuning.db',
    study_name="ucb_s5_fine_tuning",
    load_if_exists=True,
    directions=["maximize", "minimize", "minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=SEED),
)

def multi_objective(trial):
    score = objective(trial)
    attrs = trial.user_attrs
    return (
        attrs["mean"],
        attrs["std"],
        attrs["volatility"],
        attrs["drift"],
    )

study.optimize(
    multi_objective, 
    n_trials=500,
    n_jobs=1,
    show_progress_bar=True
)

