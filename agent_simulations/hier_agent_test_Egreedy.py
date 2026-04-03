from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario

import sys 
sys.path.append("C:/Users/jomon/Documents/wifi_9/csr_mab_mlo")

from pre_def_tools.pprint_dict import pprint_one_level_dict

scenario = simple_scenario_5(d_ap=20, d_sta=2, mcs=11)
total_steps = 10000

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=EGreedy, 
    agent_params_lvl1={
        "alpha": 0.4044486846101334,
        "e": 0.022387646818308484,
        "optimistic_start": 61.469386521463726
        }, 
    agent_params_lvl2={
        "alpha": 0.6099734650946088,
        "e": 0.02264440960496483,
        "optimistic_start": 0.11864219592126801
        }, 
    agent_params_lvl3={
        "alpha": 0.9997639987203092,
        "e": 0.014463140449789146,
        "optimistic_start": 5.321997157835438
        }, 
    agent_params_lvl4={
        "alpha": 0.2651533433068078,
        "e": 0.04669117425272768,
        "optimistic_start": 43.83307649265877
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

smoothed = [jnp.mean(jnp.array(throughput[i-window:i])) for i in range(window, len(throughput))]
# smoothed = jnp.array(throughput)

plt.figure(figsize=(10, 8))
plt.plot(jnp.arange(len(smoothed)), smoothed,linewidth=2)
plt.xlabel('Steps', fontsize=14)
plt.ylabel('Average Throughput (Mbps)', fontsize=14)
plt.title(f"Egreedy_4Level",  fontsize=16)
plt.legend(fontsize=12, loc="lower right", frameon=True)
plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig(f"Egreedy_4Level", dpi=600, bbox_inches='tight')
plt.show()