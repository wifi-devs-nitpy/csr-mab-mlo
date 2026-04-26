import numpy as np
import matplotlib.pyplot as plt
import jax

def plot_throughput_histogram(array: jax.Array, bin_width=50): 
        
    # Convert JAX array to NumPy array
    array = np.array(array)

    bin_width = bin_width

    # Determine maximum value
    max_value = np.max(array)

    # Create bins based on the maximum value
    bins = np.arange(0, max_value + bin_width, bin_width)

    # Plot histogram
    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(
        array,
        bins=bins,
        edgecolor='black',
        alpha=0.75
    )

    # Add frequency labels on top of each bar
    for count, patch in zip(counts, patches):
        if count > 0:  # Avoid labeling empty bins
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            plt.text(
                x,
                y,
                f'{int(count)}',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    plt.figure(figsize=(11, 7), dpi=300)

    plt.hist(
        array,
        bins=40,
        color="tab:blue",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        label="Throughput Distribution"
    )

    plt.title(
        "Distribution of Throughput",
        fontsize=18,
        fontweight="bold",
        pad=14,
    )

    ax = plt.gca()

    ax.set_xlabel(
        "Throughput [Mb/s]",
        fontsize=16,
        fontweight="bold",
        labelpad=12,
    )

    ax.set_ylabel(
        "Frequency",
        fontsize=16,
        fontweight="bold",
        labelpad=12,
    )

    # Tick styling (same as your time-series)
    ax.tick_params(axis="both", labelsize=14, width=1.4, colors="black", pad=6)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
        tick.set_color("black")

    # Grid styling
    plt.grid(axis="y", linestyle="--", alpha=0.35)

    # Optional: show mean line (adds insight instantly)
    mean_val = np.mean(array)
    plt.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {mean_val:.2f}"
    )

    plt.legend(prop={"size": 12, "weight": "semibold"})
    plt.tight_layout()
    plt.show()
    # Print additional information
    print(f"Maximum Throughput: {max_value:.2f}")
    print(f"Bin Width: {bin_width}")
    print(f"Number of Bins: {len(bins) - 1}")