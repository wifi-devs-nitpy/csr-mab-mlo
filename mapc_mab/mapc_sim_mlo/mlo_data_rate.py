from .sim import network_data_rate
from .utils import path_loss_2g, path_loss_5g, path_loss_6g
import jax
from .extension_functions import tx_power_indices_to_tx_power 
from .constants import MIN_TX_POWER, MAX_TX_POWER, CHANNEL_WIDTH_2G, CHANNEL_WIDTH_5G, CHANNEL_WIDTH_6G
import jax.numpy as jnp

def network_data_rate_mlo(
        key: jax.random.PRNGKey, 
        pos, 
        walls, 
        link_ap_sta, 
        n_tx_power_levels = 4
    ):

    n_nodes = pos.shape[0]
    
    if (walls == None):
        walls = jnp.zeros((n_nodes, n_nodes), dtype=float)

    key_2g, key_5g, key_6g = jax.random.split(key, 3)
    tx_power_array_2g = tx_power_indices_to_tx_power(link_ap_sta[0]["tx_power_indices"], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    tx_power_array_5g = tx_power_indices_to_tx_power(link_ap_sta[1]["tx_power_indices"], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    tx_power_array_6g = tx_power_indices_to_tx_power(link_ap_sta[2]["tx_power_indices"], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    
    data_rate_2g = network_data_rate(key_2g, tx=link_ap_sta[0]["tx_matrix"], pos=pos, tx_power=tx_power_array_2g, walls=walls, channel_width=CHANNEL_WIDTH_2G, path_loss_fn=path_loss_2g)
    data_rate_5g = network_data_rate(key_5g, tx=link_ap_sta[1]["tx_matrix"], pos=pos, tx_power=tx_power_array_5g, walls=walls, channel_width=CHANNEL_WIDTH_5G, path_loss_fn=path_loss_5g)
    data_rate_6g = network_data_rate(key_6g, tx=link_ap_sta[2]["tx_matrix"], pos=pos, tx_power=tx_power_array_6g, walls=walls, channel_width=CHANNEL_WIDTH_6G, path_loss_fn=path_loss_6g)

    return data_rate_2g + data_rate_5g + data_rate_6g  
