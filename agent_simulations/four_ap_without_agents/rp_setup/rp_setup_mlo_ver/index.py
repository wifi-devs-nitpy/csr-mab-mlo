import jax.numpy as jnp 

runs = jnp.stack([ jnp.asarray(list(range(10))) for i in range(10)], axis=0)
