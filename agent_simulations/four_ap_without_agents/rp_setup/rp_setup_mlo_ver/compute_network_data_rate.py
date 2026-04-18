import jax
import jax.numpy as jnp 
from mapc_mab.mapc_sim_mlo.mlo_data_rate import network_data_rate_mlo
from mapc_sim.constants import *
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
from tqdm import tqdm

def compute_throughput(ap_sta_pairs: jax.Array, d_ap: int, plot: bool = False, d_ap_sta: int = 2, link_ap_sta: tuple[list, list] = None): 
    #we are considering a scenario of 4 aps each having 4 stations arranged in a square fashion
    
    n_ap = 4
    n_sta_per_ap = 4
    n_tx_power_levels = 12

    ap_pos = jnp.array([
        [0, 0], 
        [1, 0], 
        [1, 1], 
        [0, 1]
    ]) * d_ap 
    
    dx = jnp.array([-1, 1, 1, -1]) * d_ap_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_ap_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for (x, y) in ap_pos for i in range(len(dx))]

    pos = jnp.concatenate([ap_pos, jnp.array(sta_pos)], axis=0)
    
    associations = {
        i: list(range((n_ap + i*n_sta_per_ap), (n_ap + (i+1)*n_sta_per_ap)))

        for i in range(4)
    }

    n_nodes = pos.shape[0]

    if plot == True :
        import numpy as np
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0,1, 4))
            
        plt.figure(figsize = (8, 8))

        for i, (ap, stas) in enumerate(associations.items()):
            plt.scatter(pos[ap, 0], pos[ap, 1], color=colors[i], marker='x', s=75, label=f"AP-{i}")
            plt.scatter(pos[stas, 0], pos[stas, 1], marker="o", color=colors[i], s=25)
            plt.annotate(f"AP-{i}", xy=(pos[ap, 0], pos[ap, 1] - 0.5), color=colors[i], ha='center', va='top', fontsize=15, fontweight='bold')
            for j, sta in enumerate(stas):
                plt.annotate(f"STA-{sta}", xy=(pos[sta, 0], pos[sta, 1] - 0.5), color=colors[i], ha='center', va='top', fontsize=8)
        
        plt.legend(loc='upper right', fontsize=10)
        
        plt.title("Plot of all the Nodes and Transmission", pad=20)
        plt.grid()
        plt.ylim(-5, 15)
        plt.xlim(-5, 15)
        plt.tight_layout()
        plt.show()
        

    # walls are fixed
    bss_walls_sets = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # walls between (0, 2) -> between bss0, bss2
    # walls between (0, 3) -> between bss0, bss2
    

    bss_sets_for_walls = {
        0: [0, *associations[0]], 
        1: [1, *associations[1]],
        2: [2, *associations[2]],
        3: [3, *associations[3]]
    }

    walls = jnp.zeros((n_nodes, n_nodes))

    for (x, y) in bss_walls_sets: 
        for i in bss_sets_for_walls[x]:
            for j in bss_sets_for_walls[y]: 
                walls = walls.at[i, j].set(1).at[j, i].set(1) 

    mcs = jnp.ones(n_nodes, dtype=jnp.int32) * 11
    # tx_power = jnp.ones(n_nodes) * DEFAULT_TX_POWER

    sigma = DEFAULT_SIGMA
    key = jax.random.PRNGKey(42)
    data_rate = []

    tx_matrices = jnp.zeros((3, n_nodes, n_nodes), dtype=jnp.int32)
    tx_power_indices = jnp.zeros((3, n_nodes), dtype=jnp.int32)
    ap_sta_pairs = jnp.asarray(ap_sta_pairs, dtype=jnp.int32)

    tx_matrices = tx_matrices.at[:, ap_sta_pairs[:, 0], ap_sta_pairs[:, 1]].set(1)
    tx_power_indices = tx_power_indices.at[:, :].set(7)

    link_ap_sta = (tx_matrices, tx_power_indices)

    #implementing the jit compilation 
    fast_network_data_rate = jax.jit(partial(network_data_rate_mlo, pos=pos, mcs=mcs, sigma=sigma, walls=walls, n_tx_power_levels=n_tx_power_levels))

    # first usually takes longer time -> so running it to avoid execution times in the actual runs
    fast_network_data_rate(key=key, link_ap_sta=link_ap_sta)

    for _ in range(200):
        key, run_key = jax.random.split(key, 2)
        rate = fast_network_data_rate(key=run_key, pos=pos, mcs=mcs, sigma=sigma, walls=walls, link_ap_sta=link_ap_sta)
        data_rate.append(rate)

    return jnp.asarray(data_rate).mean()


def cartesian_product(arrays: list | jax.Array):
    """
    Takes a List of arrays and returns the cartesian product of them,
    Generates all possible combinations from the elements of the arrays

    """
    grids = jnp.meshgrid(*arrays, indexing='ij')
    return jnp.stack([g.ravel() for g in grids], axis=1)

def random_search_max_throughput(
        fast_network_data_rate,
        tx_matrices,
        aps,
        key,
        n_nodes,
        n_links,
        n_tx_power_levels,
        n_samples=100_000,
        batch_size=5000,
    ):
        max_rate = 0.0

        n_batches = n_samples // batch_size

        vmapped_rate = jax.vmap(
            lambda k, tx_power_levels: fast_network_data_rate(
                key=k, link_ap_sta=(tx_matrices, tx_power_levels)
            ),
            in_axes=(0, 0),
        )

        for _ in range(n_batches):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            # Sample random TX power levels: (batch, n_ap, n_links)
            tx_power_indices_ap_links = jax.random.randint(
                subkey1,
                shape=(batch_size, aps.shape[0], n_links),
                minval=0,
                maxval=n_tx_power_levels,
            )

            # Convert → (batch, n_links, n_nodes)
            tx_power_levels_batch = jnp.zeros(
                (batch_size, n_links, n_nodes), dtype=jnp.int32
            )

            tx_power_levels_batch = tx_power_levels_batch.at[:, :, aps].set(
                jnp.swapaxes(tx_power_indices_ap_links, 1, 2)
            )

            # Keys per sample
            keys = jax.random.split(subkey2, batch_size)

            rates = vmapped_rate(keys, tx_power_levels_batch)

            max_rate = jnp.maximum(max_rate, jnp.max(rates))

        return max_rate
    


@jax.jit
def compute_max_throughput(d_ap, n_tx_power_levels=4, n_links=3):
    ap_sta_pairs = jnp.asarray([(0, 4), (1, 9), (2, 14), (3, 19)])
    aps = ap_sta_pairs[:, 0]

    list_of_links = jnp.arange(n_links, dtype=jnp.int32)
        
    n_ap = 4
    d_ap_sta = 2
    n_sta_per_ap = 4
    n_tx_power_levels = n_tx_power_levels
    
    ap_pos = jnp.array([
        [0, 0], 
        [1, 0], 
        [1, 1], 
        [0, 1]
    ]) * d_ap 

    dx = jnp.array([-1, 1, 1, -1]) * d_ap_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_ap_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for (x, y) in ap_pos for i in range(len(dx))]

    pos = jnp.concatenate([ap_pos, jnp.array(sta_pos)], axis=0)
    
    associations = {
        i: list(range((n_ap + i*n_sta_per_ap), (n_ap + (i+1)*n_sta_per_ap)))

        for i in range(4)
    }

    n_nodes = pos.shape[0]

    # walls are fixed
    bss_walls_sets = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # walls between (0, 2) -> between bss0, bss2
    # walls between (0, 3) -> between bss0, bss2
    

    bss_sets_for_walls = {
        0: [0, *associations[0]], 
        1: [1, *associations[1]],
        2: [2, *associations[2]],
        3: [3, *associations[3]]
    }

    walls = jnp.zeros((n_nodes, n_nodes))

    for (x, y) in bss_walls_sets: 
        for i in bss_sets_for_walls[x]:
            for j in bss_sets_for_walls[y]: 
                walls = walls.at[i, j].set(1).at[j, i].set(1) 

    mcs = jnp.ones(n_nodes, dtype=jnp.int32) * 11

    sigma = DEFAULT_SIGMA
    key = jax.random.PRNGKey(42)
    ap_sta_pairs = jnp.asarray(ap_sta_pairs, dtype=jnp.int32)
    aps = ap_sta_pairs[:, 0]
    stas = ap_sta_pairs[:, 1]
    
    data_rate = []
    
    #implementing the jit compilation 
    fast_network_data_rate = jax.jit(partial(network_data_rate_mlo, pos=pos, mcs=mcs, sigma=sigma, walls=walls, n_tx_power_levels=n_tx_power_levels))

    tx_matrix = jnp.zeros((n_nodes, n_nodes), dtype=jnp.int32).at[aps, stas].set(1)
    tx_matrices = jnp.repeat(tx_matrix[None, :], repeats=3, axis=0)
    
    max_rate = 0.0

    tx_levels = jnp.arange(n_tx_power_levels, dtype=jnp.int32)
    links_txp_one_ap = cartesian_product([tx_levels]*n_links)  
    aps_txp_combination = cartesian_product([jnp.arange(links_txp_one_ap.shape[0], dtype=jnp.int32)]*n_ap)
    tx_power_indices_ap_links = links_txp_one_ap[aps_txp_combination]

    n_combinations = tx_power_indices_ap_links.shape[0]

    # JIT-compile random search (treat shape/control-flow args as static)
    jit_random_search_max_throughput = jax.jit(
        random_search_max_throughput,
        static_argnames=(
            "fast_network_data_rate",
            "n_nodes",
            "n_links",
            "n_tx_power_levels",
            "n_samples",
            "batch_size",
        ),
    )

    max_rate = jit_random_search_max_throughput(
        fast_network_data_rate=fast_network_data_rate,
        tx_matrices=tx_matrices,
        aps=aps,
        key=key,
        n_nodes=n_nodes,
        n_links=n_links,
        n_tx_power_levels=n_tx_power_levels,
        n_samples=100_000,
        batch_size=10_000,
    )

    return max_rate


if __name__ == "__main__": 
    compute_throughput([(0, 4), (1, 9), (2, 14), (3, 19)], d_ap = 30, plot=False)
    print(compute_max_throughput(50))
    # run this module to plot the scenario arrangement\


"""
    Notes : 
    0. previously Improper tx matrix -> doesnt about what authors have taken
    1. main reason for the ripples last time was becuase of not averaging the throughput at particular distance over multiple steps 
    2. implement jax.jit in the compute throughput function to maximise the speed of execution - reduced to almost below 10s with each distance having (200 simulations individual simulations)
    3. This time xaxis -> log 
"""