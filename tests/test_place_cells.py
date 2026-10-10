import unittest

import numpy as np

from navigation.place_cells import PlaceCells


class PlaceCellsTests(unittest.TestCase):
    def test_normalized_gaussian_values_and_ratios(self):
        encoder = PlaceCells(centers=[[0, 0], [1, 0], [2, 0]], sigma=1.0)
        code = encoder.encode([0, 0])
        expected = np.exp([0.0, -0.5, -2.0])
        expected /= expected.sum()
        self.assertEqual(code.dtype, np.float32)
        self.assertEqual(code.shape, (3,))
        np.testing.assert_allclose(code, expected, rtol=1e-6)
        self.assertAlmostEqual(float(code[0] / code[1]), np.exp(0.5), places=6)
        self.assertAlmostEqual(float(code[1] / code[2]), np.exp(1.5), places=6)

    def test_symmetry_and_singleton_dimensions_are_preserved(self):
        for dimensions in (2, 3):
            centers = np.zeros((2, dimensions))
            centers[1, 0] = 2
            encoder = PlaceCells(centers=centers, sigma=0.5)
            for leading_shape in ((), (1,), (1, 1), (2, 1)):
                with self.subTest(dimensions=dimensions, leading_shape=leading_shape):
                    positions = np.zeros((*leading_shape, dimensions))
                    positions[..., 0] = 1
                    code = encoder.encode(positions)
                    self.assertEqual(code.shape, (*leading_shape, 2))
                    np.testing.assert_allclose(code, 0.5)
                    np.testing.assert_allclose(code.sum(axis=-1), 1.0)

    def test_narrow_fields_keep_finite_normalized_codes(self):
        encoder = PlaceCells(centers=[[0, 0], [1, 1]], sigma=0.01)
        code = encoder.encode([10, 10])
        np.testing.assert_array_equal(code, [0, 1])

    def test_sampled_centers_use_physical_bounds_and_given_rng(self):
        for bounds in ([1.5, 2.0], [1.5, 2.0, 0.75]):
            with self.subTest(bounds=bounds):
                first = PlaceCells.sample(
                    n_cells=7, bounds=bounds, sigma=0.4, rng=np.random.default_rng(31)
                )
                second = PlaceCells.sample(
                    n_cells=7, bounds=bounds, sigma=0.4, rng=np.random.default_rng(31)
                )
                self.assertEqual(first.centers.shape, (7, len(bounds)))
                self.assertEqual(first.sigma, 0.4)
                self.assertTrue(np.all(first.centers >= 0))
                self.assertTrue(np.all(first.centers <= bounds))
                np.testing.assert_array_equal(first.centers, second.centers)

    def test_invalid_encoder_inputs(self):
        invalid = (
            {"centers": [], "sigma": 1.0},
            {"centers": [[0, np.nan]], "sigma": 1.0},
            {"centers": [[0, 0]], "sigma": 0},
            {"centers": [[0, 0]], "sigma": np.nan},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises((ValueError, TypeError)):
                    PlaceCells(**arguments)
        encoder = PlaceCells(centers=[[0, 0], [1, 1]], sigma=0.5)
        for positions in ([0, 0, 0], [np.nan, 0]):
            with self.subTest(positions=positions):
                with self.assertRaises((ValueError, TypeError)):
                    encoder.encode(positions)

    def test_invalid_sampling_inputs(self):
        for arguments in ({"n_cells": 0}, {"bounds": [1, -1]}):
            with self.subTest(arguments=arguments):
                parameters = {"n_cells": 3, "bounds": [1, 1], "sigma": 0.5}
                parameters.update(arguments)
                with self.assertRaises((ValueError, TypeError)):
                    PlaceCells.sample(**parameters, rng=np.random.default_rng(5))


if __name__ == "__main__":
    unittest.main()
