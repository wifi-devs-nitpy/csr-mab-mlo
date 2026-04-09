# No need to worry about the ram now, The code is optimised with jax.vmap function
# speed increased by multiple fold with the jit compilation
# donot set the plot param to True - we are dealing with the jax.jit 

import jax.numpy as jnp 
import jax
from compute_network_data_rate import compute_throughput
import matplotlib.pyplot as plt

d = jnp.logspace(jnp.log10(4), jnp.log10(100), 200, base=10)
# d = jnp.arange(100)

compute_throughput_in_batch = jax.vmap(compute_throughput, in_axes = (None, 0, None))

# donot set the plot param to True - we are dealing with the jax.jit 

# 1 AP-STA pair
ap_station_pair_1 = [(0, 4)]
rates1 = compute_throughput_in_batch(ap_station_pair_1, d, False)
    
# 2 sta-ap pairs
ap_station_pair_2 = [(0, 4), (2, 14)]
rates2 = compute_throughput_in_batch(ap_station_pair_2, d, False)


# 3 AP-STA pairs
ap_station_pair_3 = [(0, 4), (2, 14), (3, 19)]
rates3 = compute_throughput_in_batch(ap_station_pair_3, d, False)


# 4 AP-STA pairs
ap_station_pair_4 = [(0, 4), (1, 9), (2, 14), (3, 19)]
rates4 = compute_throughput_in_batch(ap_station_pair_4, d, False)

plt.figure(figsize=(7, 6))
plt.plot(d, rates4, label='Four Aps', color = 'purple', linestyle='-')
plt.plot(d, rates3, label='Three Aps', color='teal', linestyle='-')
plt.plot(d, rates2, label='Two Aps', color='green', linestyle='-')
plt.plot(d, rates1, label='One AP', color='black', linestyle='--')
plt.xlabel('Distance d[m] ')
plt.xscale('log')
plt.xticks([10, 20, 30, 100], [10, 20, 30, 100])
plt.ylabel('Throughput [Mb/s]')
plt.legend()
plt.axvline(x=10, color="red", linestyle='--', linewidth=1)
plt.axvline(x=20, color="red", linestyle='--', linewidth=1)
plt.axvline(x=30, color="red", linestyle='--', linewidth=1)
plt.ylim(
    0,
    max(max(rates1), max(rates2), max(rates3), max(rates4) + 100)
)
plt.grid()
plt.tight_layout()
# plt.savefig('./throughput_plot_prev.png', bbox_inches='tight', dpi=1200)
plt.show()

# plt.clf()