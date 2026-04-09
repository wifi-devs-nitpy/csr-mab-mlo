import jax
import jax.numpy as jnp 
from mapc_mab.mapc_sim_mlo.sim import network_data_rate
from mapc_sim.constants import *
import matplotlib.pyplot as plt


def compute_throughput(ap_sta_pairs: jax.Array, d_ap: int, plot: bool = False, d_ap_sta: int = 2): 
    #we are considering a scenario of 4 aps each having 4 stations arranged in a square fashion
    
    n_ap = 4
    n_sta_per_ap = 4

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


    tx = jnp.zeros((n_nodes, n_nodes))

    for (ap, sta) in ap_sta_pairs: 
        tx = tx.at[ap, sta].set(1)


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
    tx_power = jnp.ones(n_nodes) * DEFAULT_TX_POWER

    sigma = DEFAULT_SIGMA
    key = jax.random.PRNGKey(42)
    data_rate = []

    #implementing the jit compilation 
    fast_network_data_rate = jax.jit(network_data_rate)

    # first usually takes longer time -> so running it to avoid execution times in the actual runs
    fast_network_data_rate(key=key, tx=tx, pos=pos, mcs=mcs, tx_power=tx_power, sigma=sigma, walls=walls)

    for _ in range(200):
        key, run_key = jax.random.split(key, 2)
        rate = fast_network_data_rate(key=run_key, tx=tx, pos=pos, mcs=mcs, tx_power=tx_power, sigma=sigma, walls=walls)
        data_rate.append(rate)

    return jnp.asarray(data_rate).mean()



if __name__ == "__main__": 
    compute_throughput([(0, 4), (1, 9), (2, 14), (3, 19)], d_ap = 30, plot=True)
    # run this module to plot the scenario arrangement\

"""
    Notes : 
    0. previously Improper tx matrix -> doesnt about what authors have taken
    1. main reason for the ripples last time was becuase of not averaging the throughput at particular distance over multiple steps 
    2. implement jax.jit in the compute throughput function to maximise the speed of execution - reduced to almost below 10s with each distance having (200 simulations individual simulations)
    3. This time xaxis -> log 
"""