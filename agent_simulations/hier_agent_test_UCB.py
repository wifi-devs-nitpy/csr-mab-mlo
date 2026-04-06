from mapc_mab.envs.static_scenarios import simple_scenario_5
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from mapc_mab.agents.mapc_agent_factory import MapcAgentFactory
from reinforced_lib.agents.mab import EGreedy, Softmax, UCB, ThompsonSampling
from tqdm import tqdm
from mapc_mab.envs.static_scenarios import StaticScenario

import sys 
# sys.path.append("C:/Users/jomon/Documents/wifi_9/csr_mab_mlo")

# from pre_def_tools.pprint_dict import pprint_one_level_dict

n_tx_power_levels: int = 12

scenario = simple_scenario_5(d_ap=40, d_sta=2, mcs=11, n_tx_power_levels=n_tx_power_levels)
total_steps = 1_00__000
agent_name = "UCB"

agent_factory = MapcAgentFactory(
    associations=scenario.associations,
    agent_type=UCB,
    agent_params_lvl1={
        "c": 2.2934036810740657,
        "gamma": 0.4102858354774459
    },
    agent_params_lvl2={
        "c": 4.1939840816259935,
        "gamma": 0.8339510447188874
    },
    agent_params_lvl3={
        "c": 2.7454017698144155,
        "gamma": 0.9854767795440708
    },
    agent_params_lvl4={
        "c": 0.5158300383662523,
        "gamma": 0.9808315444903462
    },
    tx_power_levels=n_tx_power_levels
)

agent = agent_factory.create_hierarchical_mapc_agent()


throughput = [200]

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

jnp.save(f"arrays/{agent_name}/{agent_name}_4l_{total_steps}sm_{n_tx_power_levels}txpl.npy", throughput)

for window in [10, 20, 30, 40, 50, 60, 100, 200, 400, 500, 600]:
    
    print("=" * 60)
    print(f"window size = {window}")
    print("-"*30)
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