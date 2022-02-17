import datetime as dt
import numpy as np
import typing as T
from dateutil.relativedelta import relativedelta

# import dateutil
# from dateutil import rrule
import pandas as pd
import xarray as xr

# import pytz


def datetime64_to_datetime(dt64=np.datetime64) -> dt.datetime:
    """ Convert numpy.datetime64 to dt.datetime """
    return np.datetime64(dt64, "us").astype(dt.datetime)


def datetime_to_datetime64(time=dt.datetime) -> np.datetime64:
    """ Convert dt.datetime to numpy.datetime64 """
    return np.datetime64(time)


def time_offset(
    time: dt.datetime = None,
    years: int = 0,
    months: int = 0,
    weeks: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
    **kwargs,
) -> dt.datetime:
    """
    Apply a time offset to a datetime, using calendar months, taking care of leap years.
    Accepts positive or negative offset values.

    Args:
        time: The input datetime
        years: Number of calendar years
        months: Number of calendar months
        weeks: Number of calendar weeks
        days: Number of days
        hours: Number of hours
        minutes: Number of minutes
        seconds: Number of seconds
        microseconds: Number of microseconds
        kwargs: kwargs to pass to dateutils.relativedelta.relativedelta

    Example 1: Positive monthly offset
        >>> time_offset(dt.datetime(1994, 11, 5), months=1)
        datetime.datetime(1994, 12, 5, 0, 0)

    Example 2: Negative monthly offset
        >>> time_offset(dt.datetime(1994, 11, 5), months=-1)
        datetime.datetime(1994, 10, 5, 0, 0)

    Example 3: 1 day previous
        >>> time_offset(dt.datetime(1994, 11, 5), days=-1)
        datetime.datetime(1994, 11, 4, 0, 0)

    Example 4: This time on previous day
        >>> time_offset(dt.datetime(1994, 11, 5, 7, 23), days=-1)
        datetime.datetime(1994, 11, 4, 7, 23)

    Example 5: 6 hours previous
        >>> time_offset(dt.datetime(1994, 11, 5, 7, 23), hours=-6)
        datetime.datetime(1994, 11, 5, 1, 23)

    Example 6: 1 day and 18 hours previous
        >>> time_offset(dt.datetime(1994, 11, 5, 7, 23), days=-1, hours=-18)
        datetime.datetime(1994, 11, 3, 13, 23)

    Example 7: In 27 hours time
        >>> time_offset(dt.datetime(1994, 11, 5, 7, 23), hours=27)
        datetime.datetime(1994, 11, 6, 10, 23)
    """
    return time + relativedelta(
        years=years,
        months=months,
        weeks=weeks,
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds,
        **kwargs,
    )


def get_time_from_data(
    data: T.Union[
        xr.Dataset,
        xr.DataArray,
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        T.List,
        dt.datetime,
        pd.DatetimeIndex,
    ] = None,
    time_dim: str = "time",
) -> pd.DatetimeIndex:
    """
    Returns a pd.DatetimeIndex for the values of `time_dim` in the data.

    Args:
        data: Input data which contains datetime-like objects, optionally in the column/dimension `time_dim`
        time_dim: Name of dimension that has dt.datetime objects (if required, default==`'time'`).
        Ignored for pd.Series, np.ndarray and list types.

    Returns:
        pd.DatetimeIndex of times in data
    """
    # Perform data type checks and convert data to a pandas DatetimeIndex
    if isinstance(data, (pd.Series, pd.DataFrame)):
        # Check if `time_dim` is the index
        if time_dim in data.index.names:
            times = data.index.get_level_values(time_dim)
        else:
            if isinstance(data, pd.Series):
                times = pd.DatetimeIndex(data)
            else:
                times = pd.DatetimeIndex(data[time_dim])
    elif isinstance(data, (xr.Dataset, xr.DataArray)):
        times = data.indexes[time_dim]
    elif isinstance(data, (np.ndarray, list)):
        times = pd.DatetimeIndex(data, name="time")
    elif isinstance(data, dt.datetime):
        times = pd.DatetimeIndex([data], name="time")
    elif isinstance(data, pd.DatetimeIndex):
        times = data
    else:
        data_types_str = ", ".join(
            str(i)
            for i in [
                xr.Dataset,
                xr.DataArray,
                pd.DataFrame,
                pd.Series,
                np.ndarray,
                T.List,
                dt.datetime,
                pd.DatetimeIndex,
            ]
        )
        raise TypeError(f"Data must be one of type: {data_types_str}")
    return times


def groupby_time(
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series] = None,
    time_dim: str = "time",
    grouping: str = "week",
    other_grouping_keys: T.Optional[T.Union[str, T.List[str]]] = None,
) -> T.Union[
    xr.core.groupby.DatasetGroupBy,
    xr.core.groupby.DataArrayGroupBy,
    pd.core.groupby.generic.DataFrameGroupBy,
    pd.core.groupby.generic.SeriesGroupBy,
]:
    """
    Generate a `groupby` object where data are grouped by `time` and (optionally) `id`.
    Works with Pandas Series/DataFrame as well as Xarray Dataset/DataArray.

    Args:
        data: Input data.
        time_dim: Name of index/column/dimension containing datetime-like objects.
        id_dim: Optional - Name of index/column/dimension containing unique identifiers for the points.
        grouping: Time range of grouping, one of 'week', 'month', 'dayofyear', 'dekad', 'pentad'.

    Returns:
        Groupby object (pandas or xarray type)
    """
    # Extract/Convert the `time_dim` to a pandas.DatetimeIndex
    times = get_time_from_data(data)

    if grouping == "week":
        time_groups = times.isocalendar().week
    elif grouping == "month":
        time_groups = times.month.values
    elif grouping == "year":
        time_groups = times.year.values
    elif grouping == "dayofyear":
        time_groups = get_day_of_year(times, time_dim, modify_ordinal_days=True)
    elif grouping == "dekad":
        time_groups = datetime_to_dekad_number(times)
    elif grouping == "pentad":
        time_groups = datetime_to_pentad_number(times)
    else:
        print("`grouping` must be one of 'week', 'month', 'dayofyear', 'dekad', 'pentad', 'year'")
        return None

    if isinstance(time_groups, pd.Series):
        time_groups = time_groups.values

    if isinstance(data, (xr.DataArray, xr.Dataset)):
        time_groups = xr.DataArray(time_groups, dims=time_dim, name=grouping)
    else:
        time_groups = pd.Index(data=time_groups, name=grouping)

    # Optionally group by other keys as well as `time_dim`
    if other_grouping_keys is not None:
        if isinstance(other_grouping_keys, str):
            other_grouping_keys = [other_grouping_keys]
        grouper = [time_groups] + other_grouping_keys
    else:
        grouper = time_groups

    return data.groupby(grouper)


def get_day_of_year(
    data: T.Union[
        xr.Dataset,
        xr.DataArray,
        pd.DataFrame,
        pd.Series,
        np.ndarray,
        T.List,
        dt.datetime,
        pd.DatetimeIndex,
    ] = None,
    time_dim: str = "time",
    modify_ordinal_days: bool = False,
) -> np.ndarray:
    """
    Returns array of day of year based on the times in `data`. If `modify_ordinal_days` is `True`,
    the days are modified so if a year has 366 days, both 29 February and 1 March are assigned 60.
    The returned array could be used as a 'dayofyear' variable in an xr.Dataset or pd.DataFrame
    to assist with groupby operations, fixing the issue with leap years and
    standard dt.dayofyear functionality.

    Args:
        data: Input data which contains datetime objects in `time_dim`
        time_dim: Name of dimension that has dt.datetime objects (if required, default='time').
        Ignored for pd.Series, np.ndarray and list types.
        modify_ordinal_days: If `True`, then if a year has 366 days, both 29 Feb and 1 Mar are
        assigned day of year 60.

    Returns:
        Array of 'days of year'
    """
    times = get_time_from_data(data, time_dim=time_dim)
    if modify_ordinal_days:
        march_or_later = (times.month >= 3) & times.is_leap_year
        ordinal_day = times.dayofyear.values
        modified_ordinal_day = ordinal_day
        modified_ordinal_day[march_or_later] -= 1
        return modified_ordinal_day
    else:
        return times.dayofyear.values


def datetime_to_pentad_number(
    times: T.Union[pd.DatetimeIndex, pd.Series, dt.datetime, T.List, np.ndarray] = None
) -> np.ndarray:
    """
    Determine pentad number from a datetime object, where a pentad is a group
    of 5 days, with 73 pentads in a year. Works for standard years (365 days) and
    leap years (366 days). Accepts single dt.datetime or a list/array of dt.datetime
    objects.

    Args:
        times: Datetime(s)

    Returns:
        Pentad number for the date(s)
    """
    days_of_year = get_day_of_year(times, modify_ordinal_days=True)
    pentads = np.arange(1, 366, 5)
    return np.digitize(days_of_year, bins=pentads, right=False)


def datetime_to_dekad_number(
    dates: T.Union[pd.DatetimeIndex, pd.Series, dt.datetime, np.datetime64, T.List, np.ndarray] = None
) -> np.ndarray:
    """
    Determine dekad number from a datetime object, where a dekad is a group of 10 days, with 36 dekads in a year.
    Works for standard (365 days) and leap (366 days) years. Accepts single dt.datetime or a list/array
    of dt.datetime objects.

    Args:
        dates: Date(s) to use

    Returns:
        Array of dekad number for the given dates

    Example:
        >>> datetime_to_dekad_number(np.array([np.datetime64('2004-01-01'), np.datetime64('2004-02-05')]))
        array([1, 4])
        >>> datetime_to_dekad_number([dt.datetime(2004, 1, 1), dt.datetime(2004, 1, 11), dt.datetime(2004, 2, 5)])
        array([1, 2, 4])
        >>> datetime_to_dekad_number(dt.datetime(2004, 2, 5))
        array([4])

    """
    if not isinstance(dates, pd.DatetimeIndex):
        if isinstance(dates, dt.datetime):
            dates = [dates]
        dates = pd.DatetimeIndex(dates)
    count = np.digitize(dates.day, bins=[10, 20], right=True) + 1  # Add 1 since bins start at zero
    dekads = (dates.month - 1) * 3 + count
    return dekads.values
