from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, NormalThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario

import sys 
sys.path.append("C:/Users/jomon/Documents/wifi_9/csr_mab_mlo")

from pre_def_tools.pprint_dict import pprint_one_level_dict

scenario = simple_scenario_5(d_ap=40, d_sta=2, mcs=11)
total_steps = 10000
agent_name = "Thompson_sampling"

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=NormalThompsonSampling, 
    agent_params_lvl1={
        "alpha": 6.557964055614911,
        "beta": 8.01800197877874,
        "mu": 1.8946463100294402,
        "lam": 1.0
        }, 
    agent_params_lvl2={
        "alpha": 5.271950070511999,
        "beta": 4.583729082595179,
        "mu": 3.7392639944904156,
        "lam": 1.0
        }, 
    agent_params_lvl3={
        "alpha": 9.2062507960919,
        "beta": 6.153925462013863,
        "mu": 3.1137135117793706,
        "lam": 1.0
        }, 
    agent_params_lvl4={
        "alpha": 5.353610335955899,
        "beta": 5.732358080783913,
        "mu": 3.6983478637362763,
        "lam": 1.0
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
plt.title(f"{agent_name}_4Level",  fontsize=16)
plt.legend(fontsize=12, loc="lower right", frameon=True)
plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig(f"{agent_name}_4Level", dpi=600, bbox_inches='tight')
plt.show()