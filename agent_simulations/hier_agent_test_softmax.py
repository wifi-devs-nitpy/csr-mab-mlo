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

scenario = simple_scenario_5(d_ap=40, d_sta=2, mcs=11, n_tx_power_levels=n_tx_power_levels)
total_steps = 10_000
agent_name = "Softmax"

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=Softmax,
    agent_params_lvl1={
        "lr": 0.43664735929796333,
        "tau": 0.23426581058204046,
        "alpha": 0.9695846277645586,
    },
    agent_params_lvl2={
        "lr": 2.1154290797261215,
        "tau": 7.56829206016762,
        "alpha": 0.8948273504276488,
    },
    agent_params_lvl3={
        "lr": 0.6218704727769077,
        "tau": 6.978281265126034,
        "alpha": 0.0884925020519195,
    },
    agent_params_lvl4={
        "lr": 0.03872118032174583,
        "tau": 0.12315571723666023,
        "alpha": 0.32533033076326434,
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

os.makedirs(f"arrays/{agent_name}", exist_ok=True)
jnp.save(f"arrays/{agent_name}/{agent_name}_4l_{total_steps}sm_{n_tx_power_levels}txpl.npy", throughput)
for window in [10, 20, 30, 40, 50, 60, 100, 200, 400, 500, 600]:
   
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
    plt.xlabel("Steps", fontsize=14)
    plt.ylabel("Average Throughput (Mbps)", fontsize=14)
    plt.title(f"{agent_name}_4Level (Block Avg, w={window})", fontsize=16)
    plt.legend(fontsize=12, loc="lower right", frameon=True)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.show()

   
    print("-" * 30)
    print("=" * 60)