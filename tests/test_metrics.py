import unittest

import numpy as np
import pandas as pd
import xarray as xr

from terrapyn.scoring import metrics

np.random.seed(42)
n_lat = 4
n_lon = 4
n_time = 3
data = 5 + np.random.randn(n_time, n_lat, n_lon)
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
		).values
		true_result = np.array([[0.05914716, 0.05618278], [0.07637514, 0.04562522]])
		np.testing.assert_almost_equal(result[0:2, 0:2], true_result)

	def test_mae_dataarray(self):
		result = metrics.mae_dataarray(
			model=da,
			observations=da * 0.95,
		).values
		true_result = np.array([[0.2411731, 0.23530454], [0.27400598, 0.20967361]])
		np.testing.assert_almost_equal(result[0:2, 0:2], true_result)

	def test_me_dataarray(self):
		result = metrics.me_dataarray(
			model=da,
			observations=da * 0.95,
		).values
		true_result = np.array([[0.2411731, 0.23530454], [0.27400598, 0.20967361]])
		np.testing.assert_almost_equal(result[0:2, 0:2], true_result)

	def test_rmse_dataarray(self):
		result = metrics.rmse_dataarray(
			model=da,
			observations=da * 0.95,
		).values
		true_result = np.array([[0.24320189, 0.23702908], [0.27636053, 0.21360062]])
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
		np.testing.assert_almost_equal(result.item(), 0.239735131878112)

	def test_me_dataframe(self):
		result = metrics.me_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), -0.239735131878112)

	def test_mse_dataframe(self):
		result = metrics.mse_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.059556663454582916)

	def test_rmse_dataframe(self):
		result = metrics.rmse_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.24404233947121332)

	def test_bias_dataframe(self):
		result = metrics.bias_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.95)

	def test_efficiency_dataframe(self):
		result = metrics.efficiency_df(self.df, "model", "var1")
		np.testing.assert_almost_equal(result.item(), 0.9285456087008814)

	def test_pairs_of_columns_mae(self):
		result = metrics.mae_df(self.df, ["var1", "var2"], ["var2", "var3"], output_index_names=["a", "b"])
		np.testing.assert_almost_equal(result["a"], 0.4794702637562233)

	def test_multi_column_with_single_column_mae(self):
		result = metrics.mae_df(self.df, ["var1", "var2", "var3"], "var3").values
		np.testing.assert_almost_equal(np.array([0.95894053, 0.47947026, 0.0]), result)

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
