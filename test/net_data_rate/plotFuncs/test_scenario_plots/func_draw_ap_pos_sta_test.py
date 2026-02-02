import jax
import jax.numpy as jnp
import numpy as np
import unittest

from ..plots_and_scenario_generators.random_scenarios_generator import draw_sta_positions_with_aps   # adjust import

class TestdrawapPos(unittest.TestCase):

    def test_shape_and_count(self):
        """
        Test that output shape matches number of APs + STAs.
        """
        ap_pos = jnp.array([
            [0.0, 0.0],
            [10.0, 0.0],
        ])

        associations = {
            0: [2, 3, 4],   # 3 STAs
            1: [5, 6]       # 2 STAs
        }

        pos = draw_sta_positions_with_aps(
            ap_pos=ap_pos,
            ap_sta_dist=5.0,
            associations=associations,
            seed=0
        )

        expected_n = len(ap_pos) + sum(len(v) for v in associations.values())
        self.assertEqual(pos.shape, (expected_n, 2))

    def test_ap_positions_preserved(self):
        """
        Test that AP positions remain unchanged in output.
        """
        ap_pos = jnp.array([
            [1.0, 1.0],
            [5.0, 5.0],
        ])

        associations = {
            0: [2, 3],
            1: [4]
        }

        pos = draw_sta_positions_with_aps(
            ap_pos=ap_pos,
            ap_sta_dist=3.0,
            associations=associations,
            seed=1
        )

        np.testing.assert_allclose(
            pos[:len(ap_pos)],
            ap_pos,
            rtol=0,
            atol=1e-6
        )

    def test_sta_within_distance_bounds(self):
        """
        Test that all STAs lie within the specified distance
        (square bound) from their AP.
        """
        ap_pos = jnp.array([
            [0.0, 0.0],
            [20.0, 0.0],
        ])

        associations = {
            0: [2, 3, 4],
            1: [5, 6, 7]
        }

        ap_sta_dist = 4.0

        pos = draw_sta_positions_with_aps(
            ap_pos=ap_pos,
            ap_sta_dist=ap_sta_dist,
            associations=associations,
            seed=10
        )

        offset = len(ap_pos)

        for ap_idx, sta_indices in associations.items():
            ap = ap_pos[ap_idx]
            stas = pos[offset: offset + len(sta_indices)]
            offset += len(sta_indices)

            deltas = jnp.abs(stas - ap)
            self.assertTrue(jnp.all(deltas <= ap_sta_dist + 1e-6))

    def test_reproducibility(self):
        """
        Same seed should give identical results.
        """
        ap_pos = jnp.array([
            [0.0, 0.0],
            [10.0, 10.0],
        ])

        associations = {
            0: [2, 3],
            1: [4, 5]
        }

        pos1 = draw_sta_positions_with_aps(
            ap_pos=ap_pos,
            ap_sta_dist=5.0,
            associations=associations,
            seed=123
        )

        pos2 = draw_sta_positions_with_aps(
            ap_pos=ap_pos,
            ap_sta_dist=5.0,
            associations=associations,
            seed=123
        )

        np.testing.assert_allclose(pos1, pos2)

    def test_different_seeds_produce_different_layouts(self):
        """
        Different seeds should (almost surely) produce different STA layouts.
        """
        ap_pos = jnp.array([
            [0.0, 0.0],
        ])

        associations = {
            0: [1, 2, 3, 4]
        }

        pos1 = draw_sta_positions_with_aps(
            ap_pos=ap_pos,
            ap_sta_dist=5.0,
            associations=associations,
            seed=1
        )

        pos2 = draw_sta_positions_with_aps(
            ap_pos=ap_pos,
            ap_sta_dist=5.0,
            associations=associations,
            seed=2
        )

        self.assertFalse(np.allclose(pos1, pos2))