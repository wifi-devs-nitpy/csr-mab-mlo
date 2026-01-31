# run this file from project root
# python -m test.net_data_rate.plotFuncs.test_scenario_plots.01_test_tx_scenario
import numpy as np
import sys
import os
from ..plots_and_scenario_generators.random_scenarios_generator import plot_network_scenario

# assume plot_network_scenario is already imported


def run_test(name, associations, pos, tx):
    print(f"\n=== Running test: {name} ===")
    plot_network_scenario(
        associations=associations,
        pos=pos,
        tx=tx
    )


# ---------------------------------------------------------
# 1. Single AP, single STA, simple downlink
# ---------------------------------------------------------
associations_1 = {0: [1]}
pos_1 = np.array([
    [0.0, 0.0],   # AP
    [2.0, 0.0],   # STA
])
tx_1 = np.array([
    [0, 1],
    [0, 0],
])

run_test("Single AP → Single STA", associations_1, pos_1, tx_1)


# ---------------------------------------------------------
# 2. Single AP, multiple STAs, parallel downlink
# ---------------------------------------------------------
associations_2 = {0: [1, 2, 3]}
pos_2 = np.array([
    [0.0, 0.0],   # AP
    [2.0, 1.0],
    [2.0, -1.0],
    [1.5, 0.0],
])
tx_2 = np.array([
    [0, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])

run_test("Single AP → Multiple STAs", associations_2, pos_2, tx_2)


# ---------------------------------------------------------
# 3. Two APs, isolated BSSs, no interference
# ---------------------------------------------------------
associations_3 = {
    0: [2, 3],
    1: [4, 5]
}
pos_3 = np.array([
    [0.0, 0.0],    # AP A
    [8.0, 0.0],    # AP B
    [1.0, 1.0],
    [1.0, -1.0],
    [9.0, 1.0],
    [9.0, -1.0],
])
tx_3 = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

run_test("Two isolated BSSs", associations_3, pos_3, tx_3)


# ---------------------------------------------------------
# 4. Two APs, cross-BSS interference (APs transmit)
# ---------------------------------------------------------
tx_4 = np.array([
    [0, 0, 1, 1, 1, 0],  # AP A interferes
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

run_test("Cross-BSS interference", associations_3, pos_3, tx_4)


# ---------------------------------------------------------
# 5. Uplink traffic (STAs → AP)
# ---------------------------------------------------------
tx_5 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
])

run_test("Uplink STAs → APs", associations_3, pos_3, tx_5)


# ---------------------------------------------------------
# 6. Mixed uplink + downlink in same slot
# ---------------------------------------------------------
tx_6 = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

run_test("Mixed uplink/downlink", associations_3, pos_3, tx_6)


# ---------------------------------------------------------
# 7. STA-to-STA transmission (edge case)
# ---------------------------------------------------------
tx_7 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
])

run_test("STA ↔ STA transmissions", associations_3, pos_3, tx_7)


# ---------------------------------------------------------
# 8. Dense TX matrix (stress visual clutter)
# ---------------------------------------------------------
tx_8 = np.ones((6, 6)) - np.eye(6)

run_test("Dense transmissions", associations_3, pos_3, tx_8)


# ---------------------------------------------------------
# 9. No transmissions at all
# ---------------------------------------------------------
tx_9 = np.zeros((6, 6))

run_test("No transmissions", associations_3, pos_3, tx_9)


# ---------------------------------------------------------
# 10. Three APs, asymmetric STA distribution
# ---------------------------------------------------------
associations_10 = {
    0: [3],
    1: [4, 5, 6],
    2: [7, 8]
}
pos_10 = np.array([
    [0.0, 0.0],    # AP A
    [6.0, 0.0],    # AP B
    [12.0, 0.0],   # AP C
    [1.0, 1.0],
    [7.0, 1.0],
    [6.5, -1.0],
    [5.5, 0.5],
    [13.0, 1.0],
    [11.5, -1.0],
])

tx_10 = np.array([
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1],
    *([ [0]*9 ] * 6)
])

run_test("Three BSSs, asymmetric load", associations_10, pos_10, tx_10)
