from .sim import network_data_rate
from .utils import path_loss_2g, path_loss_5g, path_loss_6g
import jax
from .extension_functions import _tx_power_indices_to_tx_power 
from .constants import MIN_TX_POWER, MAX_TX_POWER, CHANNEL_WIDTH_2G, CHANNEL_WIDTH_5G, CHANNEL_WIDTH_6G

def network_data_rate_mlo(
        key: jax.random.PRNGKey, 
        pos, 
        walls, 
        link_ap_sta, 
        n_tx_power_levels = 4
    ):

    tx_power_array_2g = _tx_power_indices_to_tx_power(link_ap_sta[0]["tx_power_indices"], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    tx_power_array_5g = _tx_power_indices_to_tx_power(link_ap_sta[1]["tx_power_indices"], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    tx_power_array_6g = _tx_power_indices_to_tx_power(link_ap_sta[2]["tx_power_indices"], n_tx_power_levels=n_tx_power_levels, min_tx_power=MIN_TX_POWER, max_tx_power=MAX_TX_POWER)
    
    data_rate_2g = network_data_rate(key, tx=link_ap_sta[0]["tx_matrix"], pos=pos, tx_power=tx_power_array_2g, walls=walls, channel_width=CHANNEL_WIDTH_2G)
    data_rate_5g = network_data_rate(key, tx=link_ap_sta[1]["tx_matrix"], pos=pos, tx_power=tx_power_array_5g, walls=walls, channel_width=CHANNEL_WIDTH_5G)
    data_rate_6g = network_data_rate(key, tx=link_ap_sta[2]["tx_matrix"], pos=pos, tx_power=tx_power_array_6g, walls=walls, channel_width=CHANNEL_WIDTH_6G)

    return data_rate_2g + data_rate_5g + data_rate_6g  
