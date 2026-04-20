from compute_network_data_rate import cartesian_product
import jax.numpy as jnp

n_links = 3
n_aps = 4

a = jnp.arange(4, dtype=jnp.int32)


link_one_ap = cartesian_product([a]*n_links)
final_combs_indices = cartesian_product([jnp.arange(link_one_ap.shape[0], dtype=jnp.int32)]*n_aps)
final_combs = link_one_ap[final_combs_indices]

print(f"Memory_size of Combinations array in Bytes: {final_combs.nbytes}")
print(f"Memory_size of Combinations array in KiB: {final_combs.nbytes / (1024)}")
print(f"Memory_size of Combinations array in MiB : {final_combs.nbytes / (1024 * 1024)}")
