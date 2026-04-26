from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

d_ap = 40

scenario = simple_scenario_5(d_ap=d_ap, d_sta=2, mcs=11, n_tx_power_levels=n_tx_power_levels)
total_steps = 5000
n_reps = 10
agent_name = "nr_ucb"

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=UCB,
    agent_params_lvl1={
        "c": 2.0,
        "gamma": 0.98
    },
    agent_params_lvl2={
        "c": 2.0,
        "gamma": 0.98
    },
    agent_params_lvl3={
        "c": 1.5,
        "gamma": 0.995
    },
    agent_params_lvl4={
        "c": 1.5,
        "gamma": 0.995
    },
    tx_power_levels=12
)


throughput = [0]
actions = []

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
        # actions.append(jnp.stack(jnp.where(link_ap_sta[0] == 1), axis=1))
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

# throughput_ema = np.asarray(ema(throughput_np, alpha=0.03))
throughput_ema = jnp.convolve(throughput, jnp.ones(100)/100, mode='same')

trials_per_second = 200
time_per_trial = 1/(trials_per_second)
x_raw = np.arange(throughput_np.shape[0]) * time_per_trial
x_ema = np.arange(throughput_ema.shape[0]) * time_per_trial

plt.figure(figsize=(11, 7), dpi=300)
plt.plot(x_raw, throughput_np, linewidth=1.2, alpha=0.25, color="tab:blue", label="Throughput (actual)")
plt.plot(x_ema, throughput_ema, linewidth=2, color="blue", label="Average Throughput")
throughput_single_link = jnp.load(f"C:/Users/studi/Desktop/Electronics/wireless_networks/wifi-9/mab_single_Link/agent_simulations/arrays/ucb/d_{d_ap}/smoothed_throughput.npy")

plt.plot(x_raw, throughput_single_link.tolist(), linewidth=1.2, color="orange", label="Throughput single link")

plt.title(
    "Hierarchical MAB (UCB) Throughput Over Time",
    fontsize=18,
    fontweight="bold",
    pad=14,
)

ax = plt.gca()
ax.axhline(y=1020, linestyle="--", linewidth=2, color="black", label="one_ap max throughput")
ax.set_xlabel("Time (s)", fontsize=16, fontweight="bold", labelpad=12, )
ax.set_ylabel("Throughput [Mb/s]", fontsize=16, fontweight="bold", labelpad=12, )
ax.tick_params(axis="both", labelsize=14, width=1.4, colors="black", pad=6)
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("bold")
    tick.set_color("black")
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend(prop={"size": 12, "weight": "semibold"})
plt.tight_layout()
plt.show()




rows1 = [{"throughput": thr.item()} for thr in throughput]
rows2 = [{"action": action.tolist()} for action in actions]

rows = [{**r1, **r2} for r1, r2 in zip(rows1, rows2)] 
import pandas as pd
df = pd.DataFrame(rows, columns=["action", "throughput"])