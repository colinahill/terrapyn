import terrapyn as tp
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr

# import calendar
# from freezegun import freeze_time
# import pytz


PACKAGE_ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PACKAGE_ROOT_DIR / "tests" / "data"

idx = pd.IndexSlice


class TestConvertDatetime64(unittest.TestCase):
    def test_object_type(self):
        result = tp.time.datetime64_to_datetime(np.datetime64("2013-04-05 07:12:34.056789"))
        self.assertEqual(result, dt.datetime(2013, 4, 5, 7, 12, 34, 56789))


class TestConvertDatetime(unittest.TestCase):
    def test_object_type(self):
        result = tp.time.datetime_to_datetime64(dt.datetime(2013, 4, 5, 7, 12, 34, 123))
        self.assertEqual(result, np.datetime64("2013-04-05 07:12:34.000123"))


class TestGetTimeFromData(unittest.TestCase):
    expected = pd.DatetimeIndex(
        ["2019-03-15", "2019-03-16", "2019-03-17"], dtype="datetime64[ns]", name="time", freq=None
    )
    df = pd.DataFrame(
        {
            "time": expected,
            "id": [123, 456, 789],
            "val": [1, 3, 5],
        }
    ).set_index(["time", "id"])

    def test_dataframe(self):
        results = tp.time.get_time_from_data(self.df.reset_index(drop=False))
        pd.testing.assert_index_equal(results, self.expected)

    def test_dataframe_time_column(self):
        results = tp.time.get_time_from_data(self.df)
        pd.testing.assert_index_equal(results, self.expected)

    def test_dataset(self):
        results = tp.time.get_time_from_data(self.df.to_xarray())
        pd.testing.assert_index_equal(results, self.expected)

    def test_list(self):
        results = tp.time.get_time_from_data(list(self.expected.to_pydatetime()))
        pd.testing.assert_index_equal(results, self.expected)

    def test_dataarray(self):
        results = tp.time.get_time_from_data(self.df.to_xarray()["val"])
        pd.testing.assert_index_equal(results, self.expected)

    def test_series_time_index(self):
        results = tp.time.get_time_from_data(self.df["val"])
        pd.testing.assert_index_equal(results, self.expected)

    def test_series_time_column(self):
        results = tp.time.get_time_from_data(pd.Series(self.expected))
        pd.testing.assert_index_equal(results, self.expected)

    def test_datetime(self):
        results = tp.time.get_time_from_data(dt.datetime(2019, 3, 15))
        pd.testing.assert_index_equal(results, pd.DatetimeIndex([dt.datetime(2019, 3, 15)], name="time"))

    def test_ndarray(self):
        results = tp.time.get_time_from_data(self.expected.to_numpy())
        pd.testing.assert_index_equal(results, self.expected)

    def test_datetimeindex(self):
        results = tp.time.get_time_from_data(self.expected)
        pd.testing.assert_index_equal(results, self.expected)

    def test_invalid_datatype(self):
        with self.assertRaises(TypeError):
            tp.time.get_time_from_data(1)


class test_groupby_time(unittest.TestCase):

    ds = xr.open_dataset(TEST_DATA_PATH / "high_resolution_365_days_test_data.nc")

    def test_dataset_groupby_week(self):
        result = tp.time.groupby_time(self.ds, grouping="week")
        self.assertEqual(result.groups, {8: [0, 1, 2], 9: [3, 4]})

    def test_dataarray_groupby_week(self):
        result = tp.time.groupby_time(self.ds["var"], grouping="week")
        self.assertEqual(result.groups, {8: [0, 1, 2], 9: [3, 4]})

    def test_dataframe_groupby_pentad(self):
        result = tp.time.groupby_time(self.ds.to_dataframe(), grouping="pentad")
        np.testing.assert_almost_equal(result.sum().values, np.array([[1788.91321445], [448.59772361]]))

    def test_series_groupby_dekad(self):
        result = tp.time.groupby_time(self.ds.to_dataframe()["var"], grouping="dekad")
        np.testing.assert_almost_equal(result.sum().values, np.array([1338.66976258, 898.84117548]))
