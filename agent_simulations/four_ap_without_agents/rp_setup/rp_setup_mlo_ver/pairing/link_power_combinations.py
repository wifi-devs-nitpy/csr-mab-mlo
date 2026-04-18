from itertools import combinations, product, chain

import jax.numpy as jnp

def cartesian_product(arrays):
    grids = jnp.meshgrid(*arrays, indexing='ij')
    return jnp.stack([g.ravel() for g in grids], axis=1)

inner = cartesian_product(
    [jnp.arange(4)] * 3
)

print(inner)