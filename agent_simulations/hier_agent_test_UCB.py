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


n_tx_power_levels: int = 10

scenario = simple_scenario_5(d_ap=10, d_sta=2, mcs=11, n_tx_power_levels=n_tx_power_levels)
total_steps = 100_000
agent_name = "UCB_sd_ch"

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=UCB,
    agent_params_lvl1={
        "c": 1.2219093406313746,
        "gamma": 0.45769673788451803
    },
    agent_params_lvl2={
        "c": 1.372388872292921,
        "gamma": 0.16732034341996938
    },
    agent_params_lvl3={
        "c": 4.64375395838126,
        "gamma": 0.6762761531313393
    },
    agent_params_lvl4={
        "c": 4.04794082305689,
        "gamma": 0.913861745934784
    },
    tx_power_levels=n_tx_power_levels
)

agent = agent_factory.create_hierarchical_mapc_agent()


throughput = [0]

key = jax.random.PRNGKey(seed=42)

# window size for smoothing

for i in tqdm(range(1, total_steps)):
    key, run_key = jax.random.split(key)
    link_ap_sta = agent.sample(throughput[-1])
    data_rate = scenario.__call__(run_key, link_ap_sta)
    # print("=" * 60)
    # print(f"at step -> prev reward = {throughput[-1]}")
    # for link in link_ap_sta.values(): 
    #     print("-" * 60)
    #     print(f"link = {link}")
    #     pprint_one_level_dict(link)
    #     print("-" * 60)
    # print("=" * 60)
    throughput.append(data_rate)

os.makedirs(f"arrays/{agent_name}", exist_ok=True)
jnp.save(f"arrays/{agent_name}/after_block_opt_1", throughput)

throughput = jnp.asarray(throughput)

throughputs_mva = jnp.convolve(throughput, jnp.ones(100) / 100, mode='valid')

# plotting actual throughputs
plt.figure(figsize=(12, 6))
plt.plot(throughput, linewidth=2, label=f"Actual Throughputs")
plt.title("Actual throughput", fontsize=14, fontweight='bold')
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

#plotting the moving average
plt.figure(figsize=(12, 6))
plt.plot(throughputs_mva, linewidth=2, label=f"MVA_100")
plt.title("Comparison of mva size=100", fontsize=14, fontweight='bold')
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


for alpha in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
    smoothed = ema(throughput, alpha=alpha)
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed, linewidth=2, label=f"EMA")
    plt.title(f"Comparison of EMA - alpha={alpha}", fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# plotting the throughput histogram
plot_throughput_histogram(throughput)

# means of throughputs after the first 1000 trails
throughput_blocks_100 = throughput.reshape(-1, 100)
means = throughput_blocks_100.mean(axis=1)
standard_deviations = throughput_blocks_100.std(axis=1)

# Convert to NumPy for faster printing and plotting
means_np = np.array(means)
stds_np = np.array(standard_deviations)

print("=" * 50)
print(f"{'Block':<10}{'Mean':<20}{'Std Dev':<20}")
print("=" * 50)

for i, (m, s) in enumerate(zip(means_np, stds_np), start=1):
    print(f"{i:<10}{m:<20.6f}{s:<20.6f}")

print("=" * 50)

# ==============================
# Plot: Block Means and Standard Deviations
# ==============================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Means
axes[0].plot(means_np, linewidth=2)
axes[0].set_title("Block Means (Block Size = 100)", fontweight='bold')
axes[0].set_ylabel("Mean Throughput")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(["Means"])

# Standard Deviations
axes[1].plot(stds_np, linewidth=2)
axes[1].set_title("Block Standard Deviations (Block Size = 100)", fontweight='bold')
axes[1].set_xlabel("Block Index")
axes[1].set_ylabel("Standard Deviation")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(["Std Dev"])

plt.tight_layout()
plt.show()