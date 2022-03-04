import unittest

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

import terrapyn as tp

"""Generate Gamma distrubuted data that could represent precipitation"""
orig_shape = 2
orig_scale = 1
orig_loc = 0  # This should always be 0 for precipitation
n_values = 360
values = np.array(
    [
        sp.stats.gamma.rvs(orig_shape, orig_loc, orig_scale, size=n_values, random_state=random_state)
        for random_state in [123, 42, 3, 14]
    ]
).reshape(2, 2, 360)

# Original values
da = xr.DataArray(
    values,
    coords={"lat": [1, 2], "lon": [3, 4], "time": pd.date_range("1980-01-01", periods=360, freq="MS")},
    name="tp",
)
series = da.isel(lon=0, lat=0).to_series()

# Gamma Function - Fitted Probability Distribution Function Parameters
ds_gamma_pdf = xr.Dataset(
    {
        "shape": (["lat", "lon"], np.array([[2.454504, 1.9035197], [1.8584378, 1.87185]])),
        "scale": (["lat", "lon"], np.array([[0.86844677, 1.0014211], [1.1162359, 0.99978834]])),
    },
    coords={"lat": [1, 2], "lon": [3, 4]},
)
array_gamma_pdf = np.array(
    [ds_gamma_pdf.isel(lon=0, lat=0)["shape"].values, ds_gamma_pdf.isel(lon=0, lat=0)["scale"].values]
)


class TestFitGammaPdf(unittest.TestCase):
    def test_dataarray(self):
        result = tp.indices.spi.fit_gamma_pdf(da)
        self.assertEqual(list(result.data_vars), ["shape", "scale"])
        np.testing.assert_array_almost_equal(
            result["shape"].values, np.array([[2.454504, 1.9035197], [1.8584378, 1.87185]])
        )
        np.testing.assert_array_almost_equal(
            result["scale"].values, np.array([[0.86844677, 1.0014211], [1.1162359, 0.99978834]])
        )

    def test_series(self):
        result = tp.indices.spi.fit_gamma_pdf(series)
        np.testing.assert_array_almost_equal(result, np.array([2.45450392, 0.86844675]))


class TestCalcGammaCdf(unittest.TestCase):
    def test_dataset(self):
        result = tp.indices.spi.calc_gamma_cdf(da, ds_gamma_pdf)
        np.testing.assert_array_almost_equal(
            result.isel(time=3).values, np.array([[0.95392674, 0.43195], [0.3511891, 0.50786465]])
        )

    def test_series(self):
        result = tp.indices.spi.calc_gamma_cdf(series, array_gamma_pdf)
        np.testing.assert_almost_equal(result.iloc[3], 0.9539267794776843)


class TestCdfToNormalPdf(unittest.TestCase):
    def test_dataset(self):
        da_gamma_cdf = tp.indices.spi.calc_gamma_cdf(da, ds_gamma_pdf)
        result = tp.indices.spi.cdf_to_normal_pdf(da_gamma_cdf)
        np.testing.assert_almost_equal(
            result.isel(time=3).values, np.array([[1.6841819, -0.17141177], [-0.38211212, 0.01971504]])
        )

    def test_series(self):
        series_gamma_cdf = tp.indices.spi.calc_gamma_cdf(series, array_gamma_pdf)
        result = tp.indices.spi.cdf_to_normal_pdf(series_gamma_cdf)
        np.testing.assert_almost_equal(result.iloc[3], 1.684182309421188)
