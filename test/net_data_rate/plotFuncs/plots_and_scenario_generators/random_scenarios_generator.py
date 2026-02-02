import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from chex import Array, Scalar
from itertools import chain

def random_scenario(
        n_ap: int = 4,
        d_ap: Scalar = 50.,
        min_sep: int = 40, 
        n_sta_per_ap: int = 10,
        d_sta: Scalar = None,
        seed: int = 1070,
):
    d_sta = 0.5*d_ap if (d_sta == None) else d_sta

    def _draw_ap_positions(key: PRNGKey,
                       n_ap: int,
                       d_ap: float,
                       min_sep: float = min_sep,
                       max_tries: int = 1000) -> Array:
        """
        Sample AP positions with a minimum separation constraint.
        """
        ap_positions = []
        key_curr = key

        for i in range(n_ap):
            for _ in range(max_tries):
                key_curr, subkey = jax.random.split(key_curr)
                candidate = jax.random.uniform(subkey, (2,)) * d_ap

                if len(ap_positions) == 0:
                    ap_positions.append([0, 0])
                    break

                dists = jnp.linalg.norm(
                    jnp.asarray(ap_positions) - candidate[None, :],
                    axis=1
                )

                if jnp.all(dists >= min_sep):
                    ap_positions.append(candidate)
                    break
            else:
                raise RuntimeError(
                    f"Could not place AP {i} with min separation {min_sep}. "
                    f"Try increasing d_ap or reducing n_ap."
                )
            
        return jnp.asarray(ap_positions)
    
    def _draw_positions(key: PRNGKey) -> Array:
        ap_key, key = jax.random.split(key)
        ap_pos = _draw_ap_positions(ap_key, n_ap, d_ap, min_sep)
        sta_pos = []

        for pos in ap_pos:
            sta_key, key = jax.random.split(key)
            center = jnp.repeat(pos[None, :], n_sta_per_ap, axis=0)
            # earlier the stations were sampled with jax.random.normal, changing it to uniform 
            stations = center + (jax.random.normal(sta_key, (n_sta_per_ap, 2)) - 0.5) * d_sta
            sta_pos += stations.tolist()

        pos = jnp.vstack([ap_pos, jnp.asarray(sta_pos)])
        return pos
    

    associations = {i: list(range((n_ap + i * n_sta_per_ap), (n_ap + (i + 1) * n_sta_per_ap))) for i in range(n_ap)}
    n_nodes = n_ap + n_ap * n_sta_per_ap
    
    tx = jnp.zeros((n_nodes, n_nodes))
    ap_sta_pairs = {ap: np.random.choice(associations[ap]).item() for ap in associations.keys()}
    tx = tx.at[np.array(list(ap_sta_pairs.keys())), np.array(list(ap_sta_pairs.values()))].set(1)
    pos = _draw_positions(jax.random.PRNGKey(seed))

    return associations, pos, tx

def draw_sta_positions_with_aps(ap_pos: Array, ap_sta_dist: Scalar, n_stas: int, seed: int = 42):
    """
    Generates random positions for stations (STAs) around each access point (AP).

    Parameters
    -----------
    ap_pos: Array
        Array of AP positions with shape (num_aps, 2).
    ap_sta_dist: Scalar
        Maximum distance from AP within which STAs can be placed.
    associations: dict[int, int]
        Dictionary mapping AP indices to the number of associated STAs.
    seed: int, optional
        Random seed for reproducibility. Defaults to 42.
    
    Returns
    ----------
        Array: Array of positions for APs and STAs with shape (num_aps + total_stas, 2).
    """
    key = jax.random.PRNGKey(seed)
    
    sta_pos = []
    for i, pos in enumerate(ap_pos):
        key, run_key = jax.random.split(key) 
        center = jnp.repeat(jnp.asarray(pos)[None, :], n_stas, axis=0)
        ap_sta_positions = center + (jax.random.uniform(run_key, (n_stas, 2), dtype=jnp.float32) - 0.5) * 2 * ap_sta_dist 
        sta_pos += ap_sta_positions.tolist()

    return jnp.concatenate([jnp.asarray(ap_pos), jnp.asarray(sta_pos)], axis=0)


def tx_matrix_generator(associations, seed=42):
    # assuming that each aps have different number of stations
    key = jax.random.PRNGKey(seed)

    n_ap = len(associations.keys())
    n_sta = len(list(chain.from_iterable(associations.values())))
    n_nodes = n_ap + n_sta

    tx_matrices = []

    for _ in range(10):   
        key, run_key = jax.random.split(key) 
        tx = jnp.zeros((n_nodes, n_nodes))
        
        ap_sta_pairs = {
                ap: jax.random.choice(run_key, jnp.asarray(associations[ap]), shape=(1,)).item() 
                for ap in associations.keys()
            }
        
        tx = tx.at[np.array(list(ap_sta_pairs.keys())), np.array(list(ap_sta_pairs.values()))].set(1)
        tx_matrices.append(tx)     

    return jnp.array(tx_matrices)