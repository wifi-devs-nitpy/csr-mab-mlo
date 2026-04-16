import numpy as np
import matplotlib.pyplot as plt
import jax

def plot_throughput_histogram(array: jax.Array, bin_width=50): 
        
    # Convert JAX array to NumPy array
    throughput_np = np.array(array)

    bin_width = bin_width

    # Determine maximum value
    max_value = np.max(throughput_np)

    # Create bins based on the maximum value
    bins = np.arange(0, max_value + bin_width, bin_width)

    # Plot histogram
    plt.figure(figsize=(12, 6))
    counts, bin_edges, patches = plt.hist(
        throughput_np,
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

    # Plot formatting
    plt.title("Histogram of Throughput", fontsize=14, fontweight='bold')
    plt.xlabel("Throughput (Mbps)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Print additional information
    print(f"Maximum Throughput: {max_value:.2f}")
    print(f"Bin Width: {bin_width}")
    print(f"Number of Bins: {len(bins) - 1}")