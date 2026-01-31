import jax.numpy as jnp
import matplotlib.pyplot as plt

from random_scenarios_generator import random_scenario, tx_matrix_generator
from scenario_plot import plot_network_scenario
from mapc_mab.mapc_sim_mlo.extension_functions import compute_throughput_for_scenario


def get_ap_sta_pairs_from_tx_matrix(tx):
    aps, stas = jnp.where(tx == 1)
    return dict(zip(aps.tolist(), stas.tolist()))


def evaluate_all_tx_matrices(associations, pos, tx_matrices, plot=False):
    """
    Compute throughput for all TX matrices of a scenario.
    Optionally plot each TX configuration.
    """
    throughputs = []

    for tx in tx_matrices:
        if plot:
            plot_network_scenario(
                associations=associations,
                pos=pos,
                tx=tx,
                figsize=(7, 7)
            )

        throughput = compute_throughput_for_scenario(tx, pos)
        throughputs.append(throughput)

        print(f"Throughput = {throughput}")
        print(get_ap_sta_pairs_from_tx_matrix(tx), "\n")

    return jnp.asarray(throughputs)


def run_single_scenario(
    n_sta_per_ap,
    n_ap=5,
    d_ap=100.0,
    d_sta=40,
    min_sep=30,
    seed=42,
    plot_layout=True,
    plot_tx=False
):

    associations, pos, _ = random_scenario(
        n_ap=n_ap,
        d_ap=d_ap,
        d_sta=d_sta,
        n_sta_per_ap=n_sta_per_ap,
        min_sep=min_sep,
        seed=seed
    )

    if plot_layout:
        print(f"\nScenario layout (n_sta_per_ap = {n_sta_per_ap})")
        plot_network_scenario(
            associations=associations,
            pos=pos,
            tx=None,
            figsize=(8, 8)
        )

    tx_matrices = tx_matrix_generator(associations)

    throughputs = evaluate_all_tx_matrices(
        associations,
        pos,
        tx_matrices,
        plot=plot_tx
    )

    avg_throughput = throughputs.mean()
    print(f"Average throughput = {avg_throughput}")
    print("-" * 60)

    return avg_throughput

# Experiment sweep

no_of_stations = [2, 4, 8, 10, 15, 20, 30]
throughput_vs_n_sta_per_ap = []

for n_sta in no_of_stations:
    avg_thr = run_single_scenario(
        n_sta_per_ap=n_sta,
        plot_layout=True,
        plot_tx=False   
    )
    throughput_vs_n_sta_per_ap.append(avg_thr)

# Plot results

plt.figure(figsize=(10, 6))
plt.plot(
    no_of_stations,
    throughput_vs_n_sta_per_ap,
    marker='o',
    linewidth=2,
    markersize=8,
    color='#1f77b4'
)

plt.xlabel('Number of Stations per AP', fontsize=12, fontweight='bold')
plt.ylabel('Average Throughput (Mbps)', fontsize=12, fontweight='bold')
plt.title('Network Throughput vs Number of Stations per Access Point',
          fontsize=14, fontweight='bold')

plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('throughput_vs_n_stations.pdf', dpi=300, bbox_inches='tight')
plt.show()
