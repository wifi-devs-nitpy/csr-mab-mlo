from mapc_mab.envs.static_scenarios import simple_scenario_5
from scenario_plot import plot_network_scenario
from mapc_mab.mapc_sim_mlo.extension_functions import compute_throughput_for_scenario
from random_scenarios_generator import draw_sta_positions_with_aps
import jax.numpy as jnp
from random_scenarios_generator import tx_matrix_generator
import matplotlib.pyplot as plt

def evaluate_all_tx_matrices(associations, pos, tx_matrices, plot=False):
    """
    Compute throughput for all TX matrices of a scenario.
    Optionally plot each TX configuration.
    """
    throughputs = []

    def get_ap_sta_pairs_from_tx_matrix(tx):
        aps, stas = jnp.where(tx == 1)
        return dict(zip(aps.tolist(), stas.tolist()))

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

        print(get_ap_sta_pairs_from_tx_matrix(tx), "\n")
        print(f"Throughput = {throughput}")

    return jnp.asarray(throughputs)

def run_single_scenario(
    associations, 
    pos,
    seed=42,
    plot_layout=True,
    plot_tx=False
):

    if plot_layout:
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


def main():
    s5 = simple_scenario_5(d_ap=50)

    associations = s5.associations
    n_ap = 4
    # n_sta = sum(len(associations[ap]) for ap in associations.keys())
    ap_positions = s5.pos[0:4, :]
    print("ap positions: ", ap_positions)

    n_stas = [4, 6, 8, 10, 12, 14, 16, 20, 24, 30]

    pos_list = [
        draw_sta_positions_with_aps(ap_pos=ap_positions, ap_sta_dist=20, n_stas=n_sta)
        for n_sta in n_stas
    ]

    avg_throughputs = []
    
    for idx, pos in enumerate(pos_list, start=1):
        print(f"Scenario {idx}: No.of stations = {n_stas[idx-1]}")
        n_sta_per_ap = n_stas[idx-1]
        associations = {i: list(range((n_ap + i * n_sta_per_ap), (n_ap + (i + 1) * n_sta_per_ap))) for i in range(n_ap)}
        print("associations : ", associations)

        avg_thr = run_single_scenario(
            associations=associations,
            pos=pos,
            plot_layout=True,
            plot_tx=False
        )
        avg_throughputs.append(avg_thr)

    print("Average throughputs for each scenario:", avg_throughputs)
    print("----" * 60)
    print("Throughput vs n_stations")
    plt.figure(figsize=(8, 5))
    plt.plot(n_stas, avg_throughputs, marker='o')
    plt.xlabel('no.of stations')
    plt.ylabel('Average Throughput')
    plt.title('Average Throughput vs. STA Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
   