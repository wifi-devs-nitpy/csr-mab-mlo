from averaging_funcs import ema 
import jax 
import jax.numpy as jnp  
import matplotlib.pyplot as plt 


throughputs = jnp.load("C:/Users/jomon/Documents/wifi_9/csr_mab_mlo/agent_simulations/arrays/ucb/UCB_4l_100000sm_12txpl.npy")

throughputs_mva = jnp.convolve(throughputs, jnp.ones(100) / 100, mode='valid')

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
    smoothed = ema(throughputs, alpha=alpha)
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed, linewidth=2, label=f"EMA")
    plt.title(f"Comparison of EMA - alpha={alpha}", fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

