# No need to worry about the ram now, The code is optimised with jax.vmap function
# speed increased by multiple fold with the jit compilation
# donot set the plot param to True - we are dealing with the jax.jit 

import jax.numpy as jnp 
import jax
from compute_network_data_rate import compute_throughput
import matplotlib.pyplot as plt
import os

d = jnp.logspace(jnp.log10(4), jnp.log10(100), 200, base=10)
# d = jnp.arange(100)

compute_throughput_in_batch = jax.vmap(compute_throughput, in_axes = (None, 0, None))

# donot set the plot param to True - we are dealing with the jax.jit 

# 1 AP-STA pair
ap_station_pair_1 = [(0, 4)]
rates1 = compute_throughput_in_batch(ap_station_pair_1, d, False)
    
# 2 sta-ap pairs
ap_station_pair_2 = [(0, 4), (2, 14)]
rates2 = compute_throughput_in_batch(ap_station_pair_2, d, False)


# 3 AP-STA pairs
ap_station_pair_3 = [(0, 4), (2, 14), (3, 19)]
rates3 = compute_throughput_in_batch(ap_station_pair_3, d, False)


# 4 AP-STA pairs
ap_station_pair_4 = [(0, 4), (1, 9), (2, 14), (3, 19)]
rates4 = compute_throughput_in_batch(ap_station_pair_4, d, False)


script_dir = os.path.dirname(os.path.abspath(__file__))
arrays_dir = os.path.join(script_dir, "arrays")
os.makedirs(arrays_dir, exist_ok=True)

jnp.save(f"{arrays_dir}/rates1.npy", rates1)
jnp.save(f"{arrays_dir}/rates2.npy", rates2)
jnp.save(f"{arrays_dir}/rates3.npy", rates3)
jnp.save(f"{arrays_dir}/rates4.npy", rates4)

plt.figure(figsize=(7, 6))
plt.plot(d, rates4, label='Four Aps', color = 'purple', linestyle='-')
plt.plot(d, rates3, label='Three Aps', color='teal', linestyle='-')
plt.plot(d, rates2, label='Two Aps', color='green', linestyle='-')
plt.plot(d, rates1, label='One AP', color='black', linestyle='--')
plt.xlabel('Distance d[m] ')
plt.xscale('log')
plt.xticks([10, 30, 50, 100], [10, 30, 50, 100])
plt.ylabel('Throughput [Mb/s]')
plt.legend()
plt.axvline(x=10, color="red", linestyle='--', linewidth=1)
plt.axvline(x=30, color="red", linestyle='--', linewidth=1)
plt.axvline(x=50, color="red", linestyle='--', linewidth=1)
plt.ylim(
    0,
    max(max(rates1), max(rates2), max(rates3), max(rates4) + 100)
)
plt.grid()
plt.tight_layout()
plt.savefig('./throughput_plot_prev.png', bbox_inches='tight', dpi=1200)
plt.show()

# Plotly version
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=d, y=rates4, mode='lines', name='Four Aps', line=dict(color='purple')))
fig.add_trace(go.Scatter(x=d, y=rates3, mode='lines', name='Three Aps', line=dict(color='teal')))
fig.add_trace(go.Scatter(x=d, y=rates2, mode='lines', name='Two Aps', line=dict(color='green')))
fig.add_trace(go.Scatter(x=d, y=rates1, mode='lines', name='One AP', line=dict(color='black', dash='dash')))

fig.add_vline(x=10, line=dict(color='red', dash='dash', width=1))
fig.add_vline(x=30, line=dict(color='red', dash='dash', width=1))
fig.add_vline(x=50, line=dict(color='red', dash='dash', width=1))

fig.update_layout(
    xaxis_title='Distance d [m]',
    yaxis_title='Throughput [Mb/s]',
    xaxis_type='log',
    xaxis=dict(tickvals=[10, 30, 50, 100], ticktext=[10, 30, 50, 100]),
    yaxis=dict(range=[0, max(max(rates1), max(rates2), max(rates3), max(rates4)) + 100]),
    legend=dict(x=0, y=1, traceorder='normal'),
    template='plotly_white',
    width=700,
    height=600
)

fig.show()

# plt.clf()

print(f"{'N_AP':<15} | {'Max_Throughput':<15}")
print("-"*32)
for x, y in [("one", max(rates1)), ("two", max(rates2)), ("three", max(rates3)), ("four", max(rates4))]:
    print(f"{x:<15} | {y:<15.4f}")