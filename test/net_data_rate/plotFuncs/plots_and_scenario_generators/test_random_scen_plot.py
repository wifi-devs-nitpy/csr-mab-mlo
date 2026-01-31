from random_scenarios_generator import random_scenario
from scenario_plot import plot_network_scenario



for seed in [1, 7, 42, 123, 999]:
    associations, pos, tx = random_scenario(
        n_ap=5,
        d_ap=100.0,
        d_sta=None,
        n_sta_per_ap=10,
        min_sep=40,
        seed=seed
    )

    plot_network_scenario(
        associations=associations,
        pos=(pos),
        tx=(tx),
        figsize=(7, 7)
    )


