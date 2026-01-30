import matplotlib.pyplot as plt
import matplotlib as mpl
import string
from copy import copy, deepcopy

import numpy as np


def ap_sta_label_to_node_index(label: str, associations: dict[int, list[int]]):
    """
    converts the label of AP or station to its node index 

    Parameters
    ----------
    label: str
        Label of the node, 
        The parameters come in the following form 
        AP -> AP-{alphabet from string.ascii_uppercase relative}
        STA -> SA-{sta index relative to the AP}        
    associations : dict[int, list[int]]
        Keys are AP indices, values are lists of STA indices associated with that AP.
    
    """
    chars = list(label.strip())
    if (chars[0] == 'A'): 
        return ord(chars[-1]) - ord('A') 
    else:
        ap_index = ord(chars[1]) - ord('A')
        return associations[ap_index][int(chars[-1])]


def plot_network_scenario(
    associations: dict[int, list[int]],
    pos: np.ndarray,
    tx: np.ndarray = None,
    figsize: tuple = (8, 8)
):
    """
    Plot a wireless network scenario with Access Points (APs) and Stations (STAs).

    Parameters
    ----------
    associations : dict[int, list[int]]
        Keys are AP indices, values are lists of STA indices associated with that AP.
    pos : np.ndarray
        Array of shape (N, 2) containing (x, y) positions of all nodes.
    show_labels : bool
        Whether to annotate node indices on the plot.
    figsize : tuple
        Size of the matplotlib figure.
    """

    n_ap = len(associations)
    bss_colors = mpl.colormaps['cool'](np.linspace(0, 1, n_ap + 2))[1:-1]
    ap_labels = string.ascii_uppercase

    fig, ax = plt.subplots(figsize=figsize)

    for i, ap in enumerate(associations):
        ax.scatter(
            pos[ap, 0], pos[ap, 1],
            marker='x',
            s=120,
            c=[bss_colors[i]],
            linewidths=3,
            label="Access Point" if i == 0 else "_nolegend_",
            zorder=4
        )

        ax.annotate(
            f"AP-{ap_labels[i]}",
            xy=(pos[ap, 0], pos[ap, 1]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
            fontweight="semibold",
            zorder=10
        )

        for j, sta in enumerate(associations[ap]):
            ax.scatter(
                pos[sta, 0], pos[sta, 1],
                marker='o',
                s=60,
                c=[bss_colors[i]],
                edgecolors="black",
                label="Station" if (i == 0 and j == 0) else "_nolegend_",
                zorder=3
            )

            ax.annotate(
                f"S{ap_labels[i]}-{j}",
                xy=(pos[sta, 0], pos[sta, 1]),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontweight="semibold",
                fontsize=10,
                zorder=10
            )

    if tx is not None:
        tx_src, tx_dst = np.where(tx == 1)

        for a, b in zip(tx_src, tx_dst):
            ax.annotate(
                "",
                xy=(pos[b, 0], pos[b, 1]),
                xytext=(pos[a, 0], pos[a, 1]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    linewidth=1.5,
                    alpha=0.8,
                    shrinkA=10,
                    shrinkB=10
                ),
                zorder=6
            )

    ax.set_title("Scenario Plot", fontsize=16, fontweight="semibold")
    ax.set_xlabel("X Position (m)", fontsize=14)
    ax.set_ylabel("Y Position (m)", fontsize=14)

    ax.tick_params(axis="both", labelsize=12, width=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
