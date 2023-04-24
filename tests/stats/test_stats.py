import unittest

import numpy as np
import pandas as pd

import terrapyn as tp


class TestSigmaClip(unittest.TestCase):
    a = np.concatenate((np.linspace(9.5, 10.5, 31), np.linspace(0, 20, 10), np.array([np.nan])))
    df = pd.DataFrame(np.column_stack([a, a - 1]), columns=["a", "b"])

    def test_value_error(self):
        with self.assertRaises(TypeError):
            tp.stats.sigma_clip(self.df, upp_sigma=3, low_sigma=3)

    def test_return_subset(self):
        result = tp.stats.sigma_clip(
            self.a, upp_sigma=3, low_sigma=3, n_iter=None, return_flags=False, return_thresholds=False
        )
        np.testing.assert_almost_equal(result[-1], 11.11111111)
        self.assertEqual(len(result), 33)

    def test_return_flags(self):
        result = tp.stats.sigma_clip(
            self.df["a"], upp_sigma=3, low_sigma=3, n_iter=None, return_flags=True, return_thresholds=False
        )
        self.assertTrue(result[34])
        self.assertEqual(result.sum(), 9)

    def test_return_flags_with_thresholds(self):
        result = tp.stats.sigma_clip(
            self.a, upp_sigma=3, low_sigma=3, n_iter=None, return_flags=True, return_thresholds=True
        )
        np.testing.assert_equal((result[0].sum(), result[1], result[2]), (9, 8.806301618952329, 11.193698381047668))

    def test_iterations_subset_pandas_series(self):
        result = tp.stats.sigma_clip(
            self.df["a"], upp_sigma=3, low_sigma=3, n_iter=2, return_flags=False, return_thresholds=False
        )
        np.testing.assert_almost_equal(result.loc[34], 6.666666666666667)

    def test_subset_with_thresholds(self):
        result = tp.stats.sigma_clip(self.a, upp_sigma=3, low_sigma=3, return_flags=False, return_thresholds=True)
        np.testing.assert_almost_equal(
            (result[0][-1], result[1], result[2]), (11.11111111111111, 8.806301618952329, 11.193698381047668)
        )
