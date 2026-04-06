from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, NormalThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario

import sys 


# from pre_def_tools.pprint_dict import pprint_one_level_dict

scenario = simple_scenario_5(d_ap=30, d_sta=2, mcs=11)
total_steps = 3000
agent_name = "softmax"

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
    hierarchical=True
)

agent = agent_factory.create_hierarchical_mapc_agent()

throughput = [0]

key = jax.random.PRNGKey(seed=42)

# window size for smoothing
window = 20

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

throughput_arr = jnp.asarray(throughput)

kernel = jnp.ones(window) / window

smoothed = jnp.convolve(throughput_arr, kernel, mode='valid')
# smoothed = jnp.array(throughput)

plt.figure(figsize=(10, 8))
plt.plot(jnp.arange(len(smoothed)), smoothed,linewidth=2, label="softmax")
plt.xlabel('Steps', fontsize=14)
plt.ylabel('Average Throughput (Mbps)', fontsize=14)
plt.title(f"{agent_name}_4Level",  fontsize=16)
plt.legend(fontsize=12, loc="lower right", frameon=True)
plt.grid(True, which='both', linestyle=':', linewidth=0.7, alpha=0.7)
plt.tight_layout()
plt.savefig(f"{agent_name}_4Level", dpi=600, bbox_inches='tight')
plt.show()