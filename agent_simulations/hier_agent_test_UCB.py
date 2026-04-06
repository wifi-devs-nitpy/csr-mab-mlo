from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario

import sys 

scenario = simple_scenario_5(d_ap=40, d_sta=2, mcs=11)
total_steps = 10_000
agent_name = "UCB"

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=UCB, 
    agent_params_lvl1={
        "c": 95.0878460790544,
        "gamma": 0.8768231620396211
        }, 
    agent_params_lvl2={
        "c": 95.0878460790544,
        "gamma": 0.8768231620396211
        }, 
    agent_params_lvl3={
        "c": 95.0878460790544,
        "gamma": 0.8768231620396211
        }, 
    agent_params_lvl4={
        "c": 95.0878460790544,
        "gamma": 0.8768231620396211
        }, 
    )

agent = agent_factory.create_hierarchical_mapc_agent()

throughput = [200]

key = jax.random.PRNGKey(seed=42)

# window size for smoothing
window = 200

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

# # smoothed = [jnp.mean(jnp.asarray(throughput[i-window:i])) for i in range(window, len(throughput))]
# smoothed_vmap_func = jax.vmap((lambda i: jnp.asarray(throughput[i-window: i]).mean()), in_axes=(0, ))
# smoothed = smoothed_vmap_func(jnp.arange(window, len(throughput)+1, 1))

# # smoothed = jnp.array(throughput)

# improving with jnp
throughput_arr = jnp.asarray(throughput)

for window in [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]: 

    print("="*60)
    print(f"window = {window}")        
    print("-"*30)
    kernel = jnp.ones(window) / window

    smoothed = jnp.convolve(throughput_arr, kernel, mode='valid')

    # ----- using the Non-overlapping window averaging ------
    # Convert to array

    # Trim so length is divisible by window
    # n = (len(throughput_arr) // window) * window
    # throughput_trimmed = throughput_arr[:n]

    # # Reshape into blocks
    # throughput_blocks = throughput_trimmed.reshape(-1, window)

    # # Take mean of each block
    # smoothed = throughput_blocks.mean(axis=1)


    plt.figure(figsize=(10, 8))
    plt.plot(jnp.arange(len(smoothed)) * window, smoothed, linewidth=2, label='UCB')
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Average Throughput (Mbps)', fontsize=14)
    plt.title(f"{agent_name}_4Level",  fontsize=16)
    plt.legend(fontsize=12, loc="lower right", frameon=True)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{agent_name}_4Level", dpi=600, bbox_inches='tight')
    plt.show()
    
    print("-"*30)
    print("="*60)
    # print(f"window = {window}")