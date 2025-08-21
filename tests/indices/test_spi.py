import unittest

import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr

import terrapyn as tp

"""Generate Gamma distrubuted data that could represent precipitation"""
orig_shape = 2
orig_scale = 1
orig_loc = 0  # This should always be 0 for precipitation
n_values = 360
values = np.array(
	[
		st.gamma.rvs(orig_shape, orig_loc, orig_scale, size=n_values, random_state=random_state)
		for random_state in [123, 42, 3, 14]
	]
).reshape(2, 2, 360)

# Original values
da = xr.DataArray(
	values,
	coords={"lat": [1, 2], "lon": [3, 4], "time": pd.date_range("1950-01-01", periods=360, freq="MS")},
	name="tp",
)
series = da.isel(lon=0, lat=0).to_series()

# Gamma Function - Fitted Probability Distribution Function Parameters
ds_gamma_pdf = xr.Dataset(
	{
		"shape": (["lat", "lon"], np.array([[2.096559, 2.17777], [1.834304, 1.898045]])),
		"scale": (["lat", "lon"], np.array([[0.973479, 0.912459], [1.126458, 0.989294]])),
	},
	coords={"lat": [1, 2], "lon": [3, 4]},
)
array_gamma_pdf = np.array(
	[ds_gamma_pdf.isel(lon=0, lat=0)["shape"].to_numpy(), ds_gamma_pdf.isel(lon=0, lat=0)["scale"].to_numpy()]
)


class TestFitGammaPdf(unittest.TestCase):
	def test_dataarray(self):
		result = tp.indices.spi.fit_gamma_pdf(da)
		self.assertEqual(list(result.data_vars), ["shape", "scale"])
		np.testing.assert_array_almost_equal(
			result["shape"].values, np.array([[2.096559, 2.17777], [1.834304, 1.898045]])
		)
		np.testing.assert_array_almost_equal(
			result["scale"].values, np.array([[0.973479, 0.912459], [1.126458, 0.989294]])
		)

	def test_invalid_data_type(self):
		with self.assertRaises(TypeError):
			tp.indices.spi.fit_gamma_pdf([1, 2, 3])

	def test_series(self):
		result = tp.indices.spi.fit_gamma_pdf(series)
		np.testing.assert_array_almost_equal(result, np.array([2.096559, 0.973479]))

	def test_dask_dataarray(self):
		result = tp.indices.spi.fit_gamma_pdf(da.chunk({"time": 10}))
		self.assertEqual(list(result.data_vars), ["shape", "scale"])
		np.testing.assert_array_almost_equal(
			result["shape"].values, np.array([[2.096559, 2.17777], [1.834304, 1.898045]])
		)
		np.testing.assert_array_almost_equal(
			result["scale"].values, np.array([[0.973479, 0.912459], [1.126458, 0.989294]])
		)


class TestCalcGammaCdf(unittest.TestCase):
	def test_dataset(self):
		result = tp.indices.spi.calc_gamma_cdf(da, ds_gamma_pdf)
		np.testing.assert_array_almost_equal(
			result.isel(time=3).values, np.array([[0.95274, 0.393063], [0.355029, 0.504819]])
		)

	def test_series(self):
		result = tp.indices.spi.calc_gamma_cdf(series, array_gamma_pdf)
		np.testing.assert_almost_equal(result.iloc[3], 0.9527398308827437)

	def test_dask_dataset(self):
		result = tp.indices.spi.calc_gamma_cdf(da.chunk({"time": 10}), ds_gamma_pdf.chunk(1))
		np.testing.assert_array_almost_equal(
			result.isel(time=3).values, np.array([[0.95274, 0.393063], [0.355029, 0.504819]])
		)

	def test_invalid_data_type(self):
		with self.assertRaises(TypeError):
			tp.indices.spi.calc_gamma_cdf([1, 2, 3], [1, 2])


class TestCdfToNormalPdf(unittest.TestCase):
	def test_dataset(self):
		da_gamma_cdf = tp.indices.spi.calc_gamma_cdf(da, ds_gamma_pdf)
		result = tp.indices.spi.cdf_to_normal_ppf(da_gamma_cdf)
		np.testing.assert_almost_equal(
			result.isel(time=3).values, np.array([[1.6720202, -0.2713455], [-0.3717769, 0.0120806]])
		)

	def test_series(self):
		series_gamma_cdf = tp.indices.spi.calc_gamma_cdf(series, array_gamma_pdf)
		result = tp.indices.spi.cdf_to_normal_ppf(series_gamma_cdf)
		np.testing.assert_almost_equal(result.iloc[3], 1.6720202033023708)

	def test_dask_dataset(self):
		da_gamma_cdf = tp.indices.spi.calc_gamma_cdf(da.chunk({"time": 10}), ds_gamma_pdf.chunk(1))
		result = tp.indices.spi.cdf_to_normal_ppf(da_gamma_cdf)
		np.testing.assert_almost_equal(
			result.isel(time=3).values, np.array([[1.6720202, -0.2713455], [-0.3717769, 0.0120806]])
		)

	def test_invalid_data_type(self):
		with self.assertRaises(TypeError):
			tp.indices.spi.cdf_to_normal_ppf([1, 2, 3])


class Test_FitGammaPdf(unittest.TestCase):
	def test_only_nan_values(self):
		result = tp.indices.spi._fit_gamma_pdf(np.full(3, np.nan))
		np.testing.assert_equal(result, np.array([np.nan, np.nan]))
