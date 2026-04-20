from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario
import os
from agent_simulations.util_funcs.averaging_funcs import ema

import sys 

n_tx_power_levels: int = 12

scenario = simple_scenario_5(d_ap=30, d_sta=2, mcs=11, n_tx_power_levels=n_tx_power_levels)
total_steps = 10_000
agent_name = "Softmax"

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


agent = agent_factory.create_hierarchical_mapc_agent()


throughput = [200]

key = jax.random.PRNGKey(seed=42)

# window size for smoothing

for i in tqdm(range(1, total_steps+1)):
    key, run_key = jax.random.split(key)
    link_ap_sta = agent.sample(throughput[-1])
    data_rate = scenario.__call__(run_key, link_ap_sta)
    throughput.append(data_rate)

# os.makedirs(f"arrays/{agent_name}", exist_ok=True)
# jnp.save(f"arrays/{agent_name}/{agent_name}_4l_{total_steps}sm_{n_tx_power_levels}txpl.npy", throughput)

# for window in [10, 20, 30, 40, 50, 60, 100, 200, 400, 500, 600]:
   
#     print("=" * 60)
#     print(f"window size = {window}")
#     print("-" * 30)

#     throughput_arr = jnp.asarray(throughput, dtype=jnp.float32)
#     kernel = jnp.ones((window,), dtype=throughput_arr.dtype) / window

#     # Equivalent to the original range(window, len(throughput)) behavior
#     smoothed = jnp.convolve(throughput_arr, kernel, mode="valid")[:-1]
#     # smoothed = jnp.array(throughput)

#     #save the array

#     plt.figure(figsize=(10, 8))
#     plt.plot(jnp.arange(len(smoothed)), smoothed,linewidth=2, label=f"{agent_name}")
#     plt.xlabel("Steps", fontsize=14)
#     plt.ylabel("Average Throughput (Mbps)", fontsize=14)
#     plt.title(f"{agent_name}_4Level (Block Avg, w={window})", fontsize=16)
#     plt.legend(fontsize=12, loc="lower right", frameon=True)
#     plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
#     plt.tight_layout()
#     plt.show()

   
#     print("-" * 30)
#     print("=" * 60)



throughput = jnp.asarray(throughput)

throughputs_mva = jnp.convolve(throughput, jnp.ones(100) / 100, mode='valid')


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


step = 100
for x in range(0, 10000, step):
    print(f"mean is {jnp.mean(throughput[x:x+step])} , std:deviation: {jnp.std(throughput[x:x+step])}")
    
print("After 2000 trails at once")
print(f"mean is {jnp.mean(throughput[2000:])} , std:deviation: {jnp.std(throughput[2000:])}")
