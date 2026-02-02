import unittest
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from reinforced_lib.agents.mab import UCB

from mapc_mab.envs.static_scenarios import simple_scenario_2, simple_scenario_5
from mapc_mab.agents import MapcAgentFactory
from mapc_mab.mapc_sim_mlo.sim import network_data_rate
from mapc_mab.mapc_sim_mlo.mlo_data_rate import network_data_rate_mlo
from mapc_mab.mapc_sim_mlo.extension_functions import tx_power_indices_to_tx_power
from mapc_mab.mapc_sim_mlo.constants import (
    MIN_TX_POWER, MAX_TX_POWER, DEFAULT_SIGMA, 
    CHANNEL_WIDTH_2G, CHANNEL_WIDTH_5G, CHANNEL_WIDTH_6G, 
    DEFAULT_TX_POWER
)


class MapcAgentFactoryTestCase(unittest.TestCase):

    def test_hierarchical_agent_4level_with_mlo(self):
        """Test 4-level hierarchical agent (with TX power control - Level 4) with MLO simulation"""
        key = jax.random.PRNGKey(42)
        np.random.seed(42)
        
        scenario = simple_scenario_5()
        
        # Create 4-level hierarchical agent with all parameters including level 4
        agent_factory = MapcAgentFactory(
            scenario.associations,
            agent_type=UCB,
            agent_params_lvl1={'c': 500.0},
            agent_params_lvl2={'c': 500.0},
            agent_params_lvl3={'c': 500.0},
            agent_params_lvl4={'c': 500.0},  # Fourth level agent for TX power
            hierarchical=True,
            tx_power_levels=4,
            n_links=3
        )
        agent = agent_factory.create_mapc_agent()

        # Simulate with MLO
        n_steps = 100
        data_rate_mlo = []
        reward = 0.
        
        # Setup MLO network data rate function
        data_rate_func_mlo = jax.jit(partial(
            network_data_rate_mlo,
            pos=scenario.pos,
            walls=scenario.walls,
            n_tx_power_levels=4
        ))

        for step in range(n_steps):
            # Sample the agent - now returns link_ap_sta dictionary
            key, sim_key = jax.random.split(key)
            link_ap_sta = agent.sample(reward=reward)

            # Simulate the MLO network with multi-link data rate
            thr = data_rate_func_mlo(key=sim_key, link_ap_sta=link_ap_sta)
            
            # Convert to reward (normalized)
            data_rate_mlo.append(float(thr))
            reward = float(thr) / 300.0  # Normalize reward


        # Plot the effective data rate for MLO
        plt.figure(figsize=(10, 6))
        plt.plot(data_rate_mlo, label='MLO Total Data Rate')
        plt.xlim(0, n_steps - 1)
        plt.xlabel('Timestep')
        plt.ylabel('Data Rate [Mb/s]')
        plt.title('MLO Hierarchical Agent (4-Level) - Scenario 5')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_5_mlo_data_rate_4level.pdf', bbox_inches='tight')
        plt.clf()
        
        # Assertions
        self.assertIsNotNone(agent)
        self.assertEqual(len(data_rate_mlo), n_steps)
        self.assertTrue(all(isinstance(d, (float, int)) for d in data_rate_mlo))

    def test_hierarchical_agent_4level_link_selection(self):
        """Test that 4-level agent correctly selects links and TX power"""
        np.random.seed(42)
        
        scenario = simple_scenario_5()
        
        agent_factory = MapcAgentFactory(
            scenario.associations,
            agent_type=UCB,
            agent_params_lvl1={'c': 500.0},
            agent_params_lvl2={'c': 500.0},
            agent_params_lvl3={'c': 500.0},
            agent_params_lvl4={'c': 500.0},
            hierarchical=True,
            tx_power_levels=4,
            n_links=3
        )
        agent = agent_factory.create_mapc_agent()

        # Sample the agent and check structure
        link_ap_sta = agent.sample(reward=0.5)

        # Verify structure
        self.assertIsInstance(link_ap_sta, dict)
        self.assertIn(0, link_ap_sta)  # 2G link
        self.assertIn(1, link_ap_sta)  # 5G link
        self.assertIn(2, link_ap_sta)  # 6G link
        
        # Check each link has required fields
        for link_id in [0, 1, 2]:
            self.assertIn('ap_sta_pairs', link_ap_sta[link_id])
            self.assertIn('tx_power_indices', link_ap_sta[link_id])
            self.assertIn('tx_matrix', link_ap_sta[link_id])
            
            # TX power indices should be in valid range
            tx_power_indices = link_ap_sta[link_id]['tx_power_indices']
            self.assertTrue(np.all(tx_power_indices < 4))
            
            # TX matrix should be binary or numeric
            tx_matrix = link_ap_sta[link_id]['tx_matrix']
            self.assertTrue(np.all((tx_matrix == 0) | (tx_matrix == 1)))

    def test_tx_power_conversion(self):
        """Test TX power indices conversion for MLO links"""
        # Create sample TX power indices
        n_nodes = 5
        tx_power_indices = np.array([0, 1, 2, 3, 1], dtype=np.int32)
        
        # Convert indices to actual TX power values
        tx_power_array = tx_power_indices_to_tx_power(
            tx_power_indices, 
            n_tx_power_levels=4,
            min_tx_power=MIN_TX_POWER,
            max_tx_power=MAX_TX_POWER
        )
        
        # Verify conversion
        self.assertEqual(len(tx_power_array), n_nodes)
        self.assertTrue(np.all(tx_power_array >= MIN_TX_POWER))
        self.assertTrue(np.all(tx_power_array <= MAX_TX_POWER))
        
        # Verify monotonicity in values
        power_values = np.linspace(MIN_TX_POWER, MAX_TX_POWER, 4)
        for i, idx in enumerate(tx_power_indices):
            self.assertAlmostEqual(tx_power_array[i], power_values[idx], places=5)


    def test_factory_agent_counts(self):
        """Test that factory creates correct number of agents for 4-level hierarchy"""
        scenario = simple_scenario_5()
        
        agent_factory = MapcAgentFactory(
            scenario.associations,
            agent_type=UCB,
            agent_params_lvl1={'c': 500.0},
            agent_params_lvl2={'c': 500.0},
            agent_params_lvl3={'c': 500.0},
            agent_params_lvl4={'c': 500.0},
            hierarchical=True,
            tx_power_levels=4,
            n_links=3
        )
        
        # Create hierarchical agent
        agent = agent_factory.create_mapc_agent()
        
        # Check agent structure
        self.assertIsNotNone(agent.find_groups_agent)
        self.assertIsNotNone(agent.assign_stations_agents)
        self.assertIsNotNone(agent.assign_links_agent)  # Level 3 - link selection
        self.assertIsNotNone(agent.select_tx_power_agent)  # Level 4 - TX power selection
        
        # Verify TX power agent is a dictionary with one entry per link
        self.assertIsInstance(agent.select_tx_power_agent, dict)
        self.assertEqual(len(agent.select_tx_power_agent), 3)  # 3 links: 2G, 5G, 6G

if __name__ == '__main__':
    unittest.main()