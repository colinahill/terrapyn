import unittest

import numpy as np
import pandas as pd
import xarray as xr

from terrapyn.scoring import grouped_scores
from terrapyn.scoring.metrics import r2_dataset


class TestGroupedScores(unittest.TestCase):
	# Generate a datframe with 3 months of daily data for 3 stations
	rng = np.random.default_rng(0)
	n = 90
	stations = ["a", "b", "c"]
	dates = pd.date_range("2020-01-01", periods=n, freq="D")
	data = rng.random((n * len(stations), 2))
	df = pd.DataFrame(
		data, index=pd.MultiIndex.from_product([dates, stations], names=["date", "id"]), columns=["tmax", "tmin"]
	)
	df["tmin_obs"] = df["tmin"] * 0.9
	df["tmax_obs"] = df["tmax"] * 0.9
	df["qc_flag"] = rng.random(len(df)) > 0.5
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
		np.testing.assert_almost_equal(result.loc[(2, "a")].values, np.array([0.0579799, 0.0553416]))

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
		np.testing.assert_almost_equal(result.loc[(2, "a", False)].values, np.array([0.0342529, 0.0502494]))

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
			np.array([0.0626408, 0.0626408, 0.0697566, 0.0587797, 0.0587797, 0.0652613]),
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
			np.array([0.0554697, 0.0554697, 0.0628578, 0.054773, 0.054773, 0.0615217]),
		)


class TestXarrayScores(unittest.TestCase):
	rng = np.random.default_rng(42)
	n_lat = 2
	n_lon = 2
	n_time = 10
	data = np.ones((n_time, n_lat, n_lon)) + np.arange(n_time)[:, None, None]
	data2 = data + rng.uniform(-0.1, 0.1, (n_time, n_lat, n_lon))
	ds = xr.Dataset(
		{
			"var": (["time", "lat", "lon"], data),
			"var2": (["time", "lat", "lon"], data2),
		},
		coords={
			"time": pd.date_range("2014-09-06", periods=n_time),
			"lat": [1, 2],
			"lon": [3, 4],
		},
	)

	def test_r2_dataset(self):
		result = r2_dataset(self.ds, "var", "var2")
		np.testing.assert_almost_equal(result, np.float64(0.9996301354890053))

	def test_r2_dataset_time_dim(self):
		result = r2_dataset(self.ds, "var", "var2", dim="time")
		np.testing.assert_almost_equal(
			result.to_numpy(), np.array([[0.99963177, 0.99969028], [0.99977611, 0.99954293]])
		)
