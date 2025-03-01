import unittest

import numpy as np
import pandas as pd
import xarray as xr

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


class TestCalculateQuantiles(unittest.TestCase):
	np.random.seed(42)
	n_lat = 10
	n_lon = 5
	n_time = 20
	data = 15 + 8 * np.random.randn(n_time, n_lat, n_lon)
	da = xr.DataArray(
		data,
		dims=["time", "lat", "lon"],
		coords={
			"time": pd.date_range("2014-09-06", periods=n_time),
			"lat": 2 + np.arange(n_lat),
			"lon": 30 + np.arange(n_lon),
		},
		name="var",
	)
	ds = da.to_dataset().chunk("auto")
	pandas_series = da.to_series()
	pandas_dataframe = da.to_dataframe()

	def test_xarray_dataset_no_rank(self):
		result = tp.stats.calculate_quantiles(self.ds, dim="time")
		np.testing.assert_almost_equal(
			result.sel(quantile=0.25).isel(lat=0, lon=0)["var"].values.item(), 9.603166819607269
		)

	def test_xarray_dataset_with_rank(self):
		result = tp.stats.calculate_quantiles(self.ds, dim="time", add_rank_coord=True)
		self.assertEqual(result.sel(quantile=0.5)["rank"].values.item(), 2)

	def test_xarray_dataarray_no_rank(self):
		result = tp.stats.calculate_quantiles(self.da, dim="time")
		np.testing.assert_almost_equal(result.sel(quantile=0.25).isel(lat=0, lon=0).values.item(), 9.603166819607269)

	def test_xarray_dataarray_with_rank(self):
		result = tp.stats.calculate_quantiles(self.da, dim="time", add_rank_coord=True)
		self.assertEqual(result.sel(quantile=0.5)["rank"].values.item(), 2)

	def test_pandas_dataframe_no_rank(self):
		result = tp.stats.calculate_quantiles(self.pandas_dataframe)
		np.testing.assert_almost_equal(result.loc[0.5].values[0], 15.202404897879106)

	def test_pandas_dataframe_wrong_axis_name(self):
		result = tp.stats.calculate_quantiles(self.pandas_dataframe, dim="time")
		np.testing.assert_almost_equal(result.loc[0.5].values[0], 15.202404897879106)

	def test_pandas_dataframe_with_rank(self):
		result = tp.stats.calculate_quantiles(self.pandas_dataframe, add_rank_coord=True)
		np.testing.assert_almost_equal(result.loc[0.5].values, np.array([15.2024049, 2.0]))

	def test_pandas_series_no_rank(self):
		result = tp.stats.calculate_quantiles(self.pandas_series)
		np.testing.assert_almost_equal(result.loc[0.5], 15.202404897879106)

	def test_pandas_series_with_rank(self):
		result = tp.stats.calculate_quantiles(self.pandas_series, add_rank_coord=True)
		np.testing.assert_almost_equal(result.loc[0.5].values, np.array([15.2024049, 2.0]))

	def test_type_error(self):
		with self.assertRaises(TypeError):
			tp.stats.calculate_quantiles([1, 2, 3])


class TestRank(unittest.TestCase):
	np.random.seed(42)
	n_lat = 10
	n_lon = 5
	n_time = 20
	data = 15 + 8 * np.random.randn(n_time, n_lat, n_lon)
	da = xr.DataArray(
		data,
		dims=["time", "lat", "lon"],
		coords={
			"time": pd.date_range("2014-09-06", periods=n_time),
			"lat": 2 + np.arange(n_lat),
			"lon": 30 + np.arange(n_lon),
		},
		name="var",
	)
	ds = da.to_dataset()
	ds_dask = ds.copy().chunk("auto")

	def test_xarray_dataset(self):
		result = tp.stats.rank(self.ds)
		np.testing.assert_equal(
			result["var"].isel(lat=3, lon=3).values,
			np.array([3, 16, 19, 10, 6, 8, 15, 13, 20, 5, 14, 2, 17, 1, 9, 4, 12, 18, 7, 11]),
		)

	def test_xarray_dataset_start_rank(self):
		result = tp.stats.rank(self.ds, starting_rank=0)
		np.testing.assert_equal(
			result["var"].isel(lat=3, lon=3).values,
			np.array([2, 15, 18, 9, 5, 7, 14, 12, 19, 4, 13, 1, 16, 0, 8, 3, 11, 17, 6, 10]),
		)

	def test_xarray_dataset_dask(self):
		result = tp.stats.rank(self.ds_dask)
		np.testing.assert_equal(
			result["var"].isel(lat=3, lon=3).values,
			np.array([3, 16, 19, 10, 6, 8, 15, 13, 20, 5, 14, 2, 17, 1, 9, 4, 12, 18, 7, 11]),
		)

	def test_xarray_dataarray(self):
		result = tp.stats.rank(self.da)
		np.testing.assert_equal(
			result.isel(lat=3, lon=3).values,
			np.array([3, 16, 19, 10, 6, 8, 15, 13, 20, 5, 14, 2, 17, 1, 9, 4, 12, 18, 7, 11]),
		)

	def test_xarray_dataarray_dask(self):
		result = tp.stats.rank(self.ds_dask["var"])
		np.testing.assert_equal(
			result.isel(lat=3, lon=3).values,
			np.array([3, 16, 19, 10, 6, 8, 15, 13, 20, 5, 14, 2, 17, 1, 9, 4, 12, 18, 7, 11]),
		)

	def test_xarray_dataset_percent(self):
		result = tp.stats.rank(self.ds, percent=True)
		np.testing.assert_equal(
			result["var"].isel(lat=3, lon=3).values,
			np.array(
				[
					0.15,
					0.8,
					0.95,
					0.5,
					0.3,
					0.4,
					0.75,
					0.65,
					1.0,
					0.25,
					0.7,
					0.1,
					0.85,
					0.05,
					0.45,
					0.2,
					0.6,
					0.9,
					0.35,
					0.55,
				]
			),
		)
