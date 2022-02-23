import datetime as dt
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytz
import xarray as xr
from freezegun import freeze_time

import terrapyn as tp

# import calendar


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

    def test_multiindex(self):
        results = tp.time.get_time_from_data(self.df.index)
        pd.testing.assert_index_equal(results, self.expected)

    def test_missing_from_multiindex(self):
        with self.assertRaises(ValueError):
            tp.time.get_time_from_data(self.df.index, time_dim="date")

    def test_invalid_datatype(self):
        with self.assertRaises(TypeError):
            tp.time.get_time_from_data(1)


class TestGroupbyTime(unittest.TestCase):

    ds = xr.open_dataset(TEST_DATA_PATH / "lat_2_lon_2_time_15_D_test_data.nc")

    def test_dataset_groupby_week(self):
        result = tp.time.groupby_time(self.ds, grouping="week")
        self.assertEqual(result.groups, {8: [0, 1, 2, 3, 4], 9: [5, 6, 7, 8, 9, 10, 11], 10: [12, 13, 14]})

    def test_dataarray_groupby_week(self):
        result = tp.time.groupby_time(self.ds["var"], grouping="week")
        self.assertEqual(result.groups, {8: [0, 1, 2, 3, 4], 9: [5, 6, 7, 8, 9, 10, 11], 10: [12, 13, 14]})

    def test_dataset_groupby_month(self):
        result = tp.time.groupby_time(self.ds, grouping="month")
        self.assertEqual(result.groups, {2: [0, 1, 2, 3, 4, 5, 6, 7, 8], 3: [9, 10, 11, 12, 13, 14]})

    def test_dataframe_groupby_pentad(self):
        result = tp.time.groupby_time(self.ds.to_dataframe(), grouping="pentad")
        np.testing.assert_almost_equal(result.sum().values, np.array([[272.59223017], [257.44398154], [295.72954042]]))

    def test_series_groupby_dekad(self):
        result = tp.time.groupby_time(self.ds.to_dataframe()["var"], grouping="dekad")
        np.testing.assert_almost_equal(result.sum().values, np.array([80.23334597, 412.85991647, 332.67248968]))

    def test_groupby_year(self):
        result = tp.time.groupby_time(self.ds, grouping="year")
        self.assertEqual(result.groups, {2019: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]})

    def test_groupby_dayofyear(self):
        result = tp.time.groupby_time(self.ds, grouping="dayofyear")
        self.assertEqual(result.groups[55], [4])

    def test_invalid_grouping(self):
        with self.assertRaises(ValueError):
            tp.time.groupby_time(self.ds, grouping="10d")

    def test_groupby_other_keys(self):
        result = tp.time.groupby_time(self.ds.to_dataframe(), grouping="year", other_grouping_keys="lon").groups[
            (2019, 30)
        ][0]
        self.assertEqual(result, (pd.Timestamp("2019-02-20 00:00:00"), 2, 30))

    def test_groupby_multiple_other_keys(self):
        result = tp.time.groupby_time(
            self.ds.to_dataframe(), grouping="year", other_grouping_keys=["lat", "lon"]
        ).groups[(2019, 2, 30)][0]
        self.assertEqual(result, (pd.Timestamp("2019-02-20 00:00:00"), 2, 30))


@freeze_time("2012-01-14 11:12:13")
class TestDailyDateRange(unittest.TestCase):
    def test_date_today(self):
        result = tp.time.daily_date_range()
        self.assertEqual(result, [dt.datetime(2012, 1, 14, 0, 0)])

    def test_delta_days(self):
        result = tp.time.daily_date_range(delta_days=-1)
        self.assertEqual(result, [dt.datetime(2012, 1, 13, 0, 0), dt.datetime(2012, 1, 14, 0, 0)])

    def test_future_dates(self):
        result = tp.time.daily_date_range(end_time=dt.datetime(2012, 1, 16), reset_time=True, hours=6)
        self.assertEqual(
            result, [dt.datetime(2012, 1, 14, 6, 0), dt.datetime(2012, 1, 15, 6, 0), dt.datetime(2012, 1, 16, 6, 0)]
        )


@freeze_time("2012-01-14 11:12:13")
class TestMonthlyDateRange(unittest.TestCase):
    def test_date_today_ignore_days(self):
        result = tp.time.monthly_date_range()
        self.assertEqual(result, [dt.datetime(2012, 1, 1, 0, 0)])

    def test_date_today_include_days(self):
        result = tp.time.monthly_date_range(reset_time=False)
        self.assertEqual(result, [dt.datetime(2012, 1, 14, 11, 12, 13)])

    def test_zero_delta_months(self):
        result = tp.time.monthly_date_range(delta_months=0)
        self.assertEqual(result, [dt.datetime(2012, 1, 1, 0, 0)])

    def test_delta_months(self):
        result = tp.time.monthly_date_range(delta_months=-2)
        self.assertEqual(
            result,
            [
                dt.datetime(2011, 11, 1, 0, 0),
                dt.datetime(2011, 12, 1, 0, 0),
                dt.datetime(2012, 1, 1, 0, 0),
            ],
        )


class TestAddDayOfYearVariable(unittest.TestCase):

    ds = xr.open_dataset(TEST_DATA_PATH / "lat_2_lon_2_time_366_D_test_data.nc")

    def test_dataset_modify_ordinal_days(self):
        result = tp.time.add_day_of_year_variable(self.ds)
        np.testing.assert_equal(result["dayofyear"].values[58:62], np.array([59, 60, 60, 61]))

    def test_dataset_no_modify_ordinal_days(self):
        result = tp.time.add_day_of_year_variable(self.ds, modify_ordinal_days=False)
        np.testing.assert_equal(result["dayofyear"].values[58:62], np.array([59, 60, 61, 62]))


class TestCheckStartEndTimeValidity(unittest.TestCase):
    def test_datetime64_with_datetime64(self):
        result = tp.time.check_start_end_time_validity(
            np.datetime64("2013-04-05 07:12:34.056789"), np.datetime64("2013-04-05 07:12:35")
        )
        self.assertTrue(result)

    def test_datetime64_with_datetime64_invalid(self):
        result = tp.time.check_start_end_time_validity(
            np.datetime64("2013-04-05 07:12:35"), np.datetime64("2013-04-05 07:12:34.056789")
        )
        self.assertFalse(result)

    def test_datetime_with_datetime64(self):
        result = tp.time.check_start_end_time_validity(
            dt.datetime(2013, 4, 5, 7, 12, 34, 56789), np.datetime64("2013-04-05 07:12:35")
        )
        self.assertTrue(result)

    @pytest.fixture(autouse=True)
    def capfd(self, capfd):
        self.capfd = capfd

    def test_verbose_warning(self):
        result = tp.time.check_start_end_time_validity(dt.datetime(2014, 1, 2), dt.datetime(2014, 1, 1), verbose=True)
        out, err = self.capfd.readouterr()
        self.assertFalse(result)
        assert out == "Warning: End time 2014-01-01 00:00:00 before start time 2014-01-02 00:00:00\n"


class TestGetDayOfYear(unittest.TestCase):

    ds = xr.open_dataset(TEST_DATA_PATH / "lat_2_lon_2_time_366_D_test_data.nc")

    def test_dataset_ordinal_days(self):
        result = tp.time.get_day_of_year(self.ds, modify_ordinal_days=False)[58:62]
        np.testing.assert_equal(result, np.array([59, 60, 61, 62]))

    def test_dataset_modify_ordinal_days(self):
        result = tp.time.get_day_of_year(self.ds, modify_ordinal_days=True)[58:62]
        np.testing.assert_equal(result, np.array([59, 60, 60, 61]))

    def test_dataarray_ordinal_days(self):
        result = tp.time.get_day_of_year(self.ds["var"], modify_ordinal_days=False)[58:62]
        np.testing.assert_equal(result, np.array([59, 60, 61, 62]))

    def test_dataframe_ordinal_days(self):
        result = tp.time.get_day_of_year(self.ds.to_dataframe(), time_dim="time", modify_ordinal_days=False)[58:62]
        np.testing.assert_equal(result, np.array([15, 15, 16, 16]))

    def test_series_ordinal_days(self):
        result = tp.time.get_day_of_year(self.ds.to_dataframe()["var"].index, modify_ordinal_days=False)[58:62]
        np.testing.assert_equal(result, np.array([15, 15, 16, 16]))

    def test_datetime_ordinal_days(self):
        result = tp.time.get_day_of_year(dt.datetime(2004, 3, 1), modify_ordinal_days=False)
        np.testing.assert_equal(result, np.array([61]))


class TestDataToLocalTime(unittest.TestCase):
    expected = pd.DatetimeIndex(
        ["2019-03-15 01:00:00", "2019-03-15 02:00:00"], dtype="datetime64[ns]", name="time", freq=None
    )
    df = pd.DataFrame(
        {"time": pd.date_range("2019-03-15", freq="h", periods=2), "id": ["a", "b"], "val": [1, 2]}
    ).set_index(["time", "id"])

    def test_dataframe(self):
        results = tp.time.data_to_local_time(self.df.copy(), "CET").index.get_level_values("time")
        pd.testing.assert_index_equal(results, self.expected)

    def test_series(self):
        results = tp.time.data_to_local_time(self.df.copy()["val"], "CET").index.get_level_values("time")
        pd.testing.assert_index_equal(results, self.expected)

    def test_dataset(self):
        results = tp.time.data_to_local_time(self.df.copy().to_xarray(), "CET").indexes["time"]
        pd.testing.assert_index_equal(results, self.expected)

    def test_dataarray(self):
        results = tp.time.data_to_local_time(self.df.copy().to_xarray(), "CET")["val"].indexes["time"]
        pd.testing.assert_index_equal(results, self.expected)

    def test_datetime(self):
        results = tp.time.data_to_local_time(dt.datetime(2019, 3, 15, 1, 0), "CET")[0]
        self.assertTrue(results, dt.datetime(2019, 3, 15, 2, 0))


class TestEnsureDatetimeIndex(unittest.TestCase):
    def test_datetime(self):
        result = tp.time._ensure_datetimeindex(dt.datetime(2021, 4, 5))
        expected = pd.DatetimeIndex(["2021-04-05 00:00:00"], dtype="datetime64[ns]", name="time", freq=None)
        pd.testing.assert_index_equal(result, expected)

    def test_list_of_datetimes(self):
        result = tp.time._ensure_datetimeindex([dt.datetime(2021, 4, 5), dt.datetime(2021, 4, 6)])
        expected = pd.DatetimeIndex(
            ["2021-04-05 00:00:00", "2021-04-06 00:00:00"], dtype="datetime64[ns]", name="time", freq=None
        )
        pd.testing.assert_index_equal(result, expected)

    def test_datetimeindex(self):
        expected = pd.DatetimeIndex(["2021-04-05 00:00:00"], dtype="datetime64[ns]", name="time", freq=None)
        result = tp.time._ensure_datetimeindex(expected)
        pd.testing.assert_index_equal(result, expected)


class TestDatetimeToUTC(unittest.TestCase):
    def test_no_timezone(self):
        result = tp.time._datetime_to_UTC(dt.datetime(2021, 4, 5))
        expected = pd.DatetimeIndex(["2021-04-05 00:00:00+00:00"], dtype="datetime64[ns, UTC]", name="time", freq=None)
        pd.testing.assert_index_equal(result, expected)

    def test_timezone_set(self):
        result = tp.time._datetime_to_UTC(dt.datetime(2021, 4, 5, tzinfo=pytz.timezone("CET")))
        expected = pd.DatetimeIndex(["2021-04-04 23:00:00+00:00"], dtype="datetime64[ns, UTC]", name="time", freq=None)
        pd.testing.assert_index_equal(result, expected)


class TestDatetimeindexToLocalTimeTzAware(unittest.TestCase):
    def test_no_timezone(self):
        result = tp.time._datetimeindex_to_local_time_tz_aware(dt.datetime(2021, 4, 5))
        expected = pd.DatetimeIndex(["2021-04-05 00:00:00+00:00"], dtype="datetime64[ns, UTC]", name="time", freq=None)
        pd.testing.assert_index_equal(result, expected)

    def test_timezone_set(self):
        result = tp.time._datetimeindex_to_local_time_tz_aware(
            dt.datetime(2021, 4, 5, tzinfo=pytz.timezone("CET")), "EST"
        )
        expected = pd.DatetimeIndex(["2021-04-04 18:00:00-05:00"], dtype="datetime64[ns, EST]", name="time", freq=None)
        pd.testing.assert_index_equal(result, expected)


class TestDatetimeindexToLocalTimeTzNaive(unittest.TestCase):
    def test_no_timezone(self):
        result = tp.time._datetimeindex_to_local_time_tz_naive(dt.datetime(2021, 4, 5))
        expected = pd.DatetimeIndex(["2021-04-05"], dtype="datetime64[ns]", name="time", freq=None)
        pd.testing.assert_index_equal(result, expected)

    def test_timezone_set(self):
        result = tp.time._datetimeindex_to_local_time_tz_naive(
            dt.datetime(2021, 4, 5, tzinfo=pytz.timezone("CET")), "EST"
        )
        expected = pd.DatetimeIndex(["2021-04-04 18:00:00"], dtype="datetime64[ns]", name="time", freq=None)
        pd.testing.assert_index_equal(result, expected)


class TestSetTimeInData(unittest.TestCase):

    df = pd.DataFrame(
        {"time": pd.date_range("2019-03-15 06:00", freq="D", periods=2), "id": ["a", "b"], "val": [1, 2]}
    ).set_index(["time", "id"])

    def test_replace_times(self):
        results = tp.time._set_time_in_data(
            self.df, new_times=pd.date_range("2021-01-1 06:00", freq="D", periods=2)
        ).index.get_level_values("time")
        expected = pd.DatetimeIndex(
            ["2021-01-01 06:00:00", "2021-01-02 06:00:00"], dtype="datetime64[ns]", name="time", freq=None
        )
        pd.testing.assert_index_equal(results, expected)

    def test_set_times_to_midnight(self):
        results = tp.time._set_time_in_data(
            self.df, set_time_to_midnight=True, hours_to_subtract=None
        ).index.get_level_values("time")
        expected = pd.DatetimeIndex(["2021-01-01", "2021-01-02"], dtype="datetime64[ns]", name="time", freq=None)
        pd.testing.assert_index_equal(results, expected)

    def test_subtract_hours(self):
        results = tp.time._set_time_in_data(
            self.df, set_time_to_midnight=True, hours_to_subtract=5
        ).index.get_level_values("time")
        expected = pd.DatetimeIndex(
            ["2020-12-31 19:00:00", "2021-01-01 19:00:00"], dtype="datetime64[ns]", name="time", freq=None
        )
        pd.testing.assert_index_equal(results, expected)


class TestUTCOffsetInHours(unittest.TestCase):
    def test_datetimeindex_no_timezone(self):
        result = tp.time.utc_offset_in_hours(pd.date_range("2019-01-02", freq="6H", periods=2), "Asia/Kolkata")
        self.assertEqual(result, 5.5)

    def test_datetime_no_timezone(self):
        result = tp.time.utc_offset_in_hours(dt.datetime(2021, 4, 5), "Asia/Kolkata")
        self.assertEqual(result, 5.5)

    def test_datetimeindex_timezone_set(self):
        result = tp.time.utc_offset_in_hours(
            pd.date_range("2019-01-02", freq="6H", periods=2, tz="CET"), "Asia/Kolkata"
        )
        self.assertEqual(result, 5.5)

    def test_datetimeindex_return_multiple_offsets(self):
        result = tp.time.utc_offset_in_hours(
            pd.date_range("2019-01-02", freq="6H", periods=2), "Asia/Kolkata", return_single_value=False
        )
        self.assertEqual(result, [5.5, 5.5])
