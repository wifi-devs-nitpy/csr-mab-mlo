from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario
from agent_simulations.util_funcs.averaging_funcs import ema
from agent_simulations.util_funcs.plot_funcs import plot_throughput_histogram
import numpy as np

import sys 
import os 


n_tx_power_levels: int = 12

total_steps = 20_000
n_reps = 1
agent_name = "nr_ucb"


class MixScen:
    def __init__(self, scenario_factory, d_sta_1: int, d_sta_2: int, max_steps: int = 100_000):
        self.scen1 = scenario_factory(d_ap=30, d_sta=d_sta_1, mcs=11, n_tx_power_levels=n_tx_power_levels)
        self.scen2 = scenario_factory(d_ap=30, d_sta=d_sta_2, mcs=11, n_tx_power_levels=n_tx_power_levels)
        self.step = 0
        self.switch_steps = [max_steps // 2]
        self.data_rate_fn1 = self.scen1.data_rate_fn
        self.data_rate_fn2 = self.scen2.data_rate_fn
        self.data_rate_fn = self.data_rate_fn1
        self.associations = self.scen1.associations

    def __call__(self, key, link_ap_sta):
        self.step += 1
        if self.step in self.switch_steps:
            self.data_rate_fn = self.data_rate_fn2 if self.data_rate_fn is self.data_rate_fn1 else self.data_rate_fn1

        return self.data_rate_fn(key, link_ap_sta=link_ap_sta)


scenario = MixScen(simple_scenario_5, 2, 4, total_steps)

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=Softmax,
    agent_params_lvl1 = {
        "lr": 0.01,
        "tau": 0.6,
        "alpha": 0.3,
    },
    agent_params_lvl2 = {
        "lr": 0.01,
        "tau": 0.8,
        "alpha": 0.4,
    },
    agent_params_lvl3 = {
        "lr": 0.01,
        "tau": 0.8,
        "alpha": 0.4,
    },
    agent_params_lvl4 = {
        "lr": 0.005,
        "tau": 0.6,
        "alpha": 0.3,
    },
    tx_power_levels=n_tx_power_levels,
)


throughput = [0]

key = jax.random.PRNGKey(seed=42)

# window size for smoothing

def run_agent(run_number: int, key, n_steps=5000):
    throughput = [0]
    agent = agent_factory.create_hierarchical_mapc_agent()
    for i in tqdm(range(1, n_steps), desc=f"run_number-{run_number}"):
        key, run_key = jax.random.split(key)
        link_ap_sta = agent.sample(throughput[-1])
        data_rate = scenario.__call__(run_key, link_ap_sta)
        throughput.append(data_rate)
    return jnp.asarray(throughput)

def compute_multiple_runs(run_numbers, keys, n_steps):
    return jnp.stack(
        [run_agent(int(run_no), run_key, n_steps) for run_no, run_key in zip(run_numbers, keys)]
    )

throughputs = compute_multiple_runs(jnp.arange(n_reps), jax.random.split(key, n_reps), total_steps)


# os.makedirs(f"arrays/{agent_name}", exist_ok=True)
# jnp.save(f"arrays/{agent_name}/{agent_name}_4l_{total_steps//1000}k-sm_{n_tx_power_levels}txpl.npy", throughput)

# throughput = jnp.asarray(throughput)

# throughputs_mva = jnp.convolve(throughput, jnp.ones(100) / 100, mode='valid')

throughput = jnp.mean(throughputs, axis=0)
throughput_np = np.asarray(throughput)

# EMA smoothing for cleaner trend visualization

throughput_ema = np.asarray(ema(throughput_np, alpha=0.03))

x_raw = np.arange(throughput_np.shape[0])
x_ema = np.arange(throughput_ema.shape[0])

plt.figure(figsize=(11, 7))
plt.plot(x_raw, throughput_np, linewidth=1.2, alpha=0.25, color="tab:blue", label="Throughput (raw)")
plt.plot(x_ema, throughput_ema, linewidth=2, color="blue", label="Throughput (EMA)")
plt.title("SoftxMax")
plt.xlabel("Step")
plt.ylabel("Throughput")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()
plt.tight_layout()
plt.show()
