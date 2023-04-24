import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import terrapyn as tp

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEST_DATA_PATH = PACKAGE_ROOT_DIR / "tests" / "data"


class TestDigitize(unittest.TestCase):
    da = xr.DataArray(
        np.array([0.7, 3, 5]).reshape(3, 1, 1),
        dims=["time", "lat", "lon"],
        coords={"time": pd.date_range("2014-09-06", periods=3), "lat": [2], "lon": [30]},
        name="var",
    )
    ds = da.to_dataset()
    ds["var2"] = ds["var"] + 3
    pandas_series = da.to_series()
    pandas_dataframe = ds.to_dataframe()
    numpy_array = da.values
    python_list = list(numpy_array)

    def test_xarray_dataset(self):
        result = tp.stats.digitize(self.ds, bins=5)
        np.testing.assert_equal(result["var"].values, np.array([[[0]], [[0]], [[1]]]))
        np.testing.assert_equal(result["var2"].values, np.array([[[0]], [[1]], [[1]]]))

    def test_xarray_dataarray(self):
        result = tp.stats.digitize(self.da, bins=5).values
        np.testing.assert_equal(result, np.array([[[0]], [[0]], [[1]]]))

    def test_dask_array(self):
        result = tp.stats.digitize(self.da.chunk(1), bins=5).values
        np.testing.assert_equal(result, np.array([[[0]], [[0]], [[1]]]))

    def test_pandas_series(self):
        result = tp.stats.digitize(self.pandas_series, bins=5).values
        np.testing.assert_equal(result, np.array([0, 0, 0]))

    def test_pandas_dataframe(self):
        result = tp.stats.digitize(self.pandas_dataframe, bins=5).values
        np.testing.assert_equal(result, np.array([[0, 0], [0, 1], [1, 1]]))

    def test_pandas_dataframe_column(self):
        result = tp.stats.digitize(self.pandas_dataframe, bins=5, columns="var").values
        np.testing.assert_equal(result, np.array([[0], [0], [1]]))

    def test_numpy_array(self):
        result = tp.stats.digitize(self.numpy_array, bins=5)
        np.testing.assert_equal(result, np.array([[[0]], [[0]], [[1]]]))

    def test_python_list(self):
        result = tp.stats.digitize(self.python_list, bins=5)
        np.testing.assert_equal(result, np.array([[[0]], [[0]], [[1]]]))

    def test_type_error(self):
        with self.assertRaises(TypeError):
            tp.stats.digitize("string", bins=[3, 6])
