from .sim import network_data_rate
from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from .constants import DEFAULT_TX_POWER, DEFAULT_SIGMA

def compute_throughput_for_scenario(tx, pos, seed=42):
    seed = 42
    key = jax.random.PRNGKey(seed)
    tx_power_array = jnp.ones((pos.shape[0],), dtype=jnp.float32) * DEFAULT_TX_POWER 
    walls = jnp.zeros(tx.shape, dtype=float)
    
    data_rate_func_fast = jax.jit(partial(network_data_rate, tx=tx, pos=pos, mcs=None, tx_power=tx_power_array, sigma=DEFAULT_SIGMA, walls=walls))

    throughput = []

    for _ in range(200):
        key, run_key = jax.random.split(key)
        thr = data_rate_func_fast(key=run_key)
        throughput.append(thr)

    return jnp.asarray(throughput).mean()
