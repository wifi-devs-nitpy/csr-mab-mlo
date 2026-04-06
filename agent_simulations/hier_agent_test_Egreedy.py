from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario

import sys 
sys.path.append(".")

scenario = simple_scenario_5(d_ap=40, d_sta=2, mcs=11)
total_steps = 2000

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=EGreedy, 
    agent_params_lvl1={
        "alpha": 0.6190443508011634,
        "e": 0.021079147434349507,
        "optimistic_start": 28.02580115407505
        }, 
    agent_params_lvl2={
        "alpha": 0.525786225102296,
        "e": 0.029119085412665096,
        "optimistic_start": 52.67521144731224
        }, 
    agent_params_lvl3={
        "alpha": 0.808941375014292,
        "e": 0.03003810096861446,
        "optimistic_start": 65.24736631881609
        }, 
    agent_params_lvl4={
        "alpha": 0.8252966772746064,
        "e": 0.059353379674557366,
        "optimistic_start": 18.235559629882363
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

smoothed = [jnp.mean(jnp.asarray(throughput[0:i])) for i in range(1, len(throughput))]
# smoothed = jnp.array(throughput)

plt.figure(figsize=(10, 8))
plt.plot(jnp.arange(len(smoothed)), smoothed, linewidth=2)
plt.xlabel('Steps', fontsize=14)
plt.ylabel('Average Throughput (Mbps)', fontsize=14)
plt.title(f"Egreedy_4Level",  fontsize=16)
plt.legend(fontsize=12, loc="lower right", frameon=True)
plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig(f"Egreedy_4Level", dpi=600, bbox_inches='tight')
plt.show()