import numpy as np
from itertools import product 
import jax.numpy as jnp


# a = np.array([1, 2, 3, 1, 5])
# mask = np.vstack(a, a)


# links = np.array(list(product(range(2), repeat=3)))

a = jnp.arange(0, 4, 1)
print(list(product((a.tolist()), repeat=2)))


X, Y, Z = jnp.meshgrid(*(jnp.repeat(a[None, :], repeats=3, axis=0)), indexing='ij')
b = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
print(b.tolist())
