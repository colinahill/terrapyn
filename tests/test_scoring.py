import unittest

import numpy as np
import pandas as pd

from terrapyn.scoring import grouped_scores


class TestGroupedScores(unittest.TestCase):
    # Generate a datframe with 3 months of daily data for 3 stations
    np.random.seed(0)
    n = 90
    stations = ["a", "b", "c"]
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    data = np.random.rand(n * len(stations), 2)
    df = pd.DataFrame(
        data, index=pd.MultiIndex.from_product([dates, stations], names=["date", "id"]), columns=["tmax", "tmin"]
    )
    df["tmin_obs"] = df["tmin"] * 0.9
    df["tmax_obs"] = df["tmax"] * 0.9
    df["qc_flag"] = np.random.rand(len(df)) > 0.5
    model_names = ["tmax", "tmin"]
    obs_names = ["tmax_obs", "tmin_obs"]

    def test_single_metric_with_id(self):
        result = grouped_scores(
            self.df,
            metrics="mae",
            groupby_time=True,
            time_dim="date",
            time_grouping="month",
            other_grouping_keys=["id"],
            model_names=self.model_names,
            obs_names=self.obs_names,
            output_index_names=self.model_names,
        )
        np.testing.assert_almost_equal(result.loc[(2, "a")].values, np.array([0.05231232, 0.04902575]))

    def test_single_metric_with_multiple_grouping_keys(self):
        result = grouped_scores(
            self.df,
            metrics="me",
            groupby_time=True,
            time_dim="date",
            time_grouping="week",
            other_grouping_keys=["id", "qc_flag"],
            model_names=self.model_names,
            obs_names=self.obs_names,
            output_index_names=self.model_names,
        )
        np.testing.assert_almost_equal(result.loc[(2, "a", False)].values, np.array([0.05943933, 0.03105201]))

    def test_multiple_metrics_with_multiple_grouping_keys(self):
        result = grouped_scores(
            self.df,
            metrics=["me", "mae", "rmse"],
            groupby_time=True,
            time_dim="date",
            time_grouping="month",
            other_grouping_keys=["id", "qc_flag"],
            model_names=self.model_names,
            obs_names=self.obs_names,
            output_index_names=self.model_names,
        )
        np.testing.assert_almost_equal(
            result.loc[(2, "a", False)].values,
            np.array([0.04544406, 0.04544406, 0.05455041, 0.04328315, 0.04328315, 0.05421829]),
        )

    def test_no_time_multiple_metrics_with_multiple_grouping_keys(self):
        result = grouped_scores(
            self.df,
            metrics=["me", "mae", "rmse"],
            groupby_time=False,
            other_grouping_keys=["id", "qc_flag"],
            model_names=self.model_names,
            obs_names=self.obs_names,
            output_index_names=self.model_names,
        )
        np.testing.assert_almost_equal(
            result.loc[("a", False)].values,
            np.array([0.04780516, 0.04780516, 0.0564465, 0.04926192, 0.04926192, 0.05689823]),
        )
