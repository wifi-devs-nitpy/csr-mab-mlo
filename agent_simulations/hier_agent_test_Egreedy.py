from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario
import os

import sys 

n_tx_power_levels: int = 12

scenario = simple_scenario_5(d_ap=30, d_sta=2, mcs=11, n_tx_power_levels=n_tx_power_levels)
total_steps = 5000
agent_name = "EGreedy"

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=EGreedy,
    agent_params_lvl1={
        "e": 0.021079147434349507,
        "optimistic_start": 28.02580115407505,
        "alpha": 0.6190443508011634
    },
    agent_params_lvl2={
        "e": 0.029119085412665096,
        "optimistic_start": 52.67521144731224,
        "alpha": 0.525786225102296
    },
    agent_params_lvl3={
        "e": 0.03003810096861446,
        "optimistic_start": 65.24736631881609,
        "alpha": 0.808941375014292
    },
    agent_params_lvl4={
        "e": 0.059353379674557366,
        "optimistic_start": 18.235559629882363,
        "alpha": 0.8252966772746064
    },
    tx_power_levels=n_tx_power_levels
)

agent = agent_factory.create_hierarchical_mapc_agent()


throughput = [200]

key = jax.random.PRNGKey(seed=42)

# window size for smoothing

for i in tqdm(range(1, total_steps+1)):
    key, run_key = jax.random.split(key)
    link_ap_sta = agent.sample(throughput[-1])
    data_rate = scenario.batch__call__(run_key, link_ap_sta)
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
jnp.save(f"arrays/{agent_name}/{agent_name}_4l_{total_steps}sm_{n_tx_power_levels}txpl.npy", throughput)

for window in [10, 20, 30, 40, 50, 60, 100, 200, 400, 500]:
    print("=" * 60)
    print(f"window size = {window}")
    print("-" * 30)
    
    throughput_arr = jnp.asarray(throughput, dtype=jnp.float32)
    kernel = jnp.ones((window,), dtype=throughput_arr.dtype) / window

    # Equivalent to the original range(window, len(throughput)) behavior
    smoothed = jnp.convolve(throughput_arr, kernel, mode="valid")[:-1]
    # smoothed = jnp.array(throughput)

    #save the array

    plt.figure(figsize=(10, 8))
    plt.plot(jnp.arange(len(smoothed)), smoothed,linewidth=2, label=f"{agent_name}")
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Average Throughput (Mbps)', fontsize=14)
    plt.title(f"{agent_name}_4Level",  fontsize=16)
    plt.legend(fontsize=12, loc="lower right", frameon=True)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{agent_name}_4Level", dpi=600, bbox_inches='tight')
    plt.show()
    
    print("-"*30)
    print("=" * 60)