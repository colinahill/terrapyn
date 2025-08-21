import unittest

import numpy as np
import pandas as pd
import xarray as xr

from terrapyn.scoring import metrics

rng = np.random.default_rng(42)
n_lat = 4
n_lon = 4
n_time = 3
data = 5 + rng.standard_normal((n_time, n_lat, n_lon))
da = xr.DataArray(
	data,
	dims=["time", "lat", "lon"],
	coords={
		"time": pd.date_range("2014-09-06", periods=n_time),
		"lat": 3 + np.arange(n_lat),
		"lon": 13 + np.arange(n_lon),
	},
	name="var",
)
ds = da.to_dataset()


class TestXarrayFunctions(unittest.TestCase):
	"""
	Test the scoring functions that accept xarray.Dataset and xarray.DataArray
	"""

	def test_bias_dataarray(self):
		result = metrics.bias_dataarray(
			model=da,
			observations=da * 0.95,
		)
		true_result = 1.0526315789473686
		self.assertEqual(result, true_result)

	def test_mse_dataarray(self):
		result = metrics.mse_dataarray(
			model=da,
			observations=da * 0.95,
		).to_numpy()
		true_result = np.array([[0.0642529, 0.0412807], [0.0469627, 0.0413605]])
		np.testing.assert_almost_equal(result[0:2, 0:2], true_result)

	def test_mae_dataarray(self):
		result = metrics.mae_dataarray(
			model=da,
			observations=da * 0.95,
		).to_numpy()
		true_result = np.array([[0.2526871, 0.2031227], [0.2125026, 0.2029456]])
		np.testing.assert_almost_equal(result[0:2, 0:2], true_result)

	def test_me_dataarray(self):
		result = metrics.me_dataarray(
			model=da,
			observations=da * 0.95,
		).to_numpy()
		true_result = np.array([[0.2526871, 0.2031227], [0.2125026, 0.2029456]])
		np.testing.assert_almost_equal(result[0:2, 0:2], true_result)

	def test_rmse_dataarray(self):
		result = metrics.rmse_dataarray(
			model=da,
			observations=da * 0.95,
		).to_numpy()
		true_result = np.array([[0.2534815, 0.2031766], [0.2167088, 0.2033727]])
		np.testing.assert_almost_equal(result[0:2, 0:2], true_result)


class TestDataFrameFunctions(unittest.TestCase):
	"""
	Test the scoring functions that accept pandas.DataFrame
	"""

	df = ds.to_dataframe().rename(columns={"var": "var1"})
	df["var2"] = df["var1"] * 0.9
	df["var3"] = df["var1"] * 0.8
	df["var4"] = df["var1"] * 0.7
	df["model"] = df["var1"] * 0.95

	def test_mae_dataframe(self):
		result = metrics.mae_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.2539729780125937)

	def test_me_dataframe(self):
		result = metrics.me_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), -0.2539729780125937)

	def test_mse_dataframe(self):
		result = metrics.mse_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.06599027687757043)

	def test_rmse_dataframe(self):
		result = metrics.rmse_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.2568857272749314)

	def test_bias_dataframe(self):
		result = metrics.bias_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.95)

	def test_efficiency_dataframe(self):
		result = metrics.efficiency_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.8891294862647221)

	def test_pairs_of_columns_mae(self):
		result = metrics.mae_df(self.df, ["var1", "var2"], ["var2", "var3"], output_index_names=["a", "b"])
		np.testing.assert_almost_equal(result["a"], 0.5079459560251869)

	def test_multi_column_with_single_column_mae(self):
		result = metrics.mae_df(self.df, ["var1", "var2", "var3"], "var3").to_numpy()
		np.testing.assert_almost_equal(np.array([1.015892, 0.507946, 0.0]), result)

	def test_pairs_of_columns_bias(self):
		result = metrics.bias_df(self.df, ["var1", "var2"], ["var2", "var3"], output_index_names=["a", "b"])
		np.testing.assert_almost_equal(np.array([1.11111111, 1.125]), result.loc[["a", "b"]].values)

	def test_multi_column_with_single_column_bias(self):
		result = metrics.bias_df(self.df, ["var1", "var2", "var3"], "var4")
		np.testing.assert_almost_equal(np.array([1.42857143, 1.28571429, 1.14285714]), result.values)

	def test_wrong_dimensions(self):
		df2 = self.df.iloc[:1]
		with self.assertRaises(ValueError):
			metrics.mean_error(self.df, df2)
