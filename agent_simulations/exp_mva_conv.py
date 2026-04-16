from agent_simulations.util_funcs.averaging_funcs import ema 
from agent_simulations.util_funcs.plot_funcs import plot_throughput_histogram
import numpy as np 
import jax 
import jax.numpy as jnp  
import matplotlib.pyplot as plt 


throughput = jnp.load("./arrays/ucb/UCB_4l_100000sm_12txpl.npy")

throughputs_mva = jnp.convolve(throughput, jnp.ones(10) / 10, mode='valid')

plt.figure(figsize=(12, 6))
plt.plot(throughputs_mva, linewidth=2, label=f"MVA_100")
plt.title("Comparison of mva size=100", fontsize=14, fontweight='bold')
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Value", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

for alpha in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]:
    smoothed = ema(throughput, alpha=alpha)
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed, linewidth=2, label=f"EMA")
    plt.title(f"Comparison of EMA - alpha={alpha}", fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# plotting the throughput histogram
plot_throughput_histogram(throughput)

# means of throughputs after the first 1000 trails
throughput_blocks_100 = throughput.reshape(-1, 100)
means = throughput_blocks_100.mean(axis=1)
standard_deviations = throughput_blocks_100.std(axis=1)

# Convert to NumPy for faster printing and plotting
means_np = np.array(means)
stds_np = np.array(standard_deviations)

print("=" * 50)
print(f"{'Block':<10}{'Mean':<20}{'Std Dev':<20}")
print("=" * 50)

for i, (m, s) in enumerate(zip(means_np, stds_np), start=1):
    print(f"{i:<10}{m:<20.6f}{s:<20.6f}")

print("=" * 50)

# ==============================
# Plot: Block Means and Standard Deviations
# ==============================
fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Means
axes[0].plot(means_np, linewidth=2)
axes[0].set_title("Block Means (Block Size = 100)", fontweight='bold')
axes[0].set_ylabel("Mean Throughput")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend(["Means"])

# Standard Deviations
axes[1].plot(stds_np, linewidth=2)
axes[1].set_title("Block Standard Deviations (Block Size = 100)", fontweight='bold')
axes[1].set_xlabel("Block Index")
axes[1].set_ylabel("Standard Deviation")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend(["Std Dev"])

plt.tight_layout()
plt.show()