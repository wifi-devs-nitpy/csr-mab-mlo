import numpy as np
from ..plots_and_scenario_generators.scenario_plot import plot_network_scenario
# Import the function you wrote
# from your_module import plot_network_scenario
import numpy as np

associations = {
    0: [3, 4],
    1: [5, 6, 7],
    2: [8]
}

pos = np.array([
    [0.0, 0.0],
    [4.0, 0.0],
    [2.0, 3.0],
    [0.5, 0.8],
    [-0.6, 1.2],
    [3.6, 0.7],
    [4.5, 1.1],
    [3.2, -0.4],
    [2.1, 2.4],
])

plot_network_scenario(associations, pos)