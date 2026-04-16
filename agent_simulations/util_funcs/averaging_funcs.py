import jax
import jax.numpy as jnp
from jax import lax

@jax.jit
def ema(data, alpha=0.05):
    def step(prev, x):
        new = alpha * x + (1.0 - alpha) * prev
        return new, new

    init = data[0]
    _, ema_vals = lax.scan(step, init, data[1:])
    return jnp.concatenate([jnp.array([init]), ema_vals])


@jax.jit
def cumulative_average(data):
    cumsum = jnp.cumsum(data)
    steps = jnp.arange(1, data.shape[0] + 1)
    return cumsum / steps


@jax.jit
def moving_average(data, window):
    kernel = jnp.ones(window) / window
    return jnp.convolve(data, kernel, mode='valid')


batched_ema = jax.jit(jax.vmap(ema, in_axes=(0, None)))