import datetime as dt
import typing as T

import numpy as np
import pandas as pd

try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

import xarray as xr
from dateutil import rrule
from dateutil.relativedelta import relativedelta

from . import utils

# import dateutil


def datetime64_to_datetime(dt64=np.datetime64) -> dt.datetime:
    """Convert numpy.datetime64 to dt.datetime"""
    return np.datetime64(dt64, "us").astype(dt.datetime)


def datetime_to_datetime64(time=dt.datetime) -> np.datetime64:
    """Convert dt.datetime to numpy.datetime64"""
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
        pd.MultiIndex,
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
    elif isinstance(data, pd.MultiIndex):
        # Check if `time_dim` is the index
        if time_dim in data.names:
            times = data.get_level_values(time_dim)
        else:
            raise ValueError(f"`time_dim` of {time_dim} not found in pd.Multindex")
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
        raise TypeError(f"Data is of type {type(data)} but must be one of type: {data_types_str}")
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
        grouping: Time range of grouping, one of 'week', 'month', 'dayofyear', 'dekad', 'pentad'.
        other_grouping_keys: (Optional, only for pd.Dataframe/pd.Series) Other keys to use to group
        the data, in addition to the time grouping.

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
        raise ValueError("`grouping` must be one of 'week', 'month', 'dayofyear', 'dekad', 'pentad', 'year'")

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
    if isinstance(dates, (dt.datetime, np.datetime64)):
        dates = [dates]
    dates = pd.DatetimeIndex(dates)
    count = np.digitize(dates.day, bins=[10, 20], right=True) + 1  # Add 1 since bins start at zero
    dekads = (dates.month - 1) * 3 + count
    return dekads.values


def daily_date_range(
    start_time: dt.datetime = None,
    end_time: dt.datetime = None,
    delta_days: int = None,
    hours: T.Union[int, T.List[int]] = None,
    reset_time: bool = True,
    ref_hour: int = 0,
    ref_minutes: int = 0,
    ref_seconds: int = 0,
    ref_microseconds: int = 0,
) -> T.List[dt.datetime]:
    """
    Generate a list of dates with a daily frequency, where a datetime objects is generated for each
    given hour in `hours`. If `start_time` or `end_time` is `None` today's date is used.
    If `delta_days` is given, a range of dates are generated using this day offset and the `start_time`.
    Otherwise, the range is between `start_time` and `end_time`. If `reset_time==True`, the hours, minutes,
    seconds and microseconds are replaced with the given reference values `ref_hour` etc.

    Args:
        start_time: Start date for range
        end_time: End date for range
        delta_days: The number of days to offset the `start_time`, where the generated date range
        is from this offset date to the `start_time`. Can be positive or negative.
        hours: The hours that will be generated for each day, such that multiple datetimes will be
        generated with the same day value, and where the hour are those given in the `hours` lsit
        reset_time: If `True`, the hours, minutes, seconds and microseconds are replaced with
        the given reference values `ref_hour` etc. This should be `True` if you want include all
        days, even if the day is not complete

    Returns:
        List of dates

    Example 1: Daily date range with hours, minutes and seconds reset to 0

        >>> start_time = dt.datetime(1994, 11, 5, 7, 23)
        >>> end_time = dt.datetime(1994, 11, 7, 0)
        >>> daily_date_range(start_time, end_time)  # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(1994, 11, 5, 0, 0),
         datetime.datetime(1994, 11, 6, 0, 0),
         datetime.datetime(1994, 11, 7, 0, 0)]

    Example 2: Daily date range with a specific hour

        >>> start_time = dt.datetime(1994, 11, 5, 7, 23)
        >>> end_time = dt.datetime(1994, 11, 7, 0)
        >>> daily_date_range(start_time, end_time, hours=3)  # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(1994, 11, 5, 3, 0),
         datetime.datetime(1994, 11, 6, 3, 0),
         datetime.datetime(1994, 11, 7, 3, 0)]

    Example 3: Daily date range with multiple hours

        >>> start_time = dt.datetime(1994, 11, 5, 7, 23)
        >>> end_time = dt.datetime(1994, 11, 7, 0)
        >>> daily_date_range(start_time, end_time, hours=[3, 6])  # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(1994, 11, 5, 3, 0),
         datetime.datetime(1994, 11, 5, 6, 0),
         datetime.datetime(1994, 11, 6, 3, 0),
         datetime.datetime(1994, 11, 6, 6, 0),
         datetime.datetime(1994, 11, 7, 3, 0),
         datetime.datetime(1994, 11, 7, 6, 0)]

    Example 4: Delta days range, only taking the day into account

        >>> date = dt.datetime(1994, 11, 5, 7, 23)
        >>> daily_date_range(date, delta_days=-1, hours=[3, 9])  # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(1994, 11, 4, 3, 0),
         datetime.datetime(1994, 11, 4, 9, 0),
         datetime.datetime(1994, 11, 5, 3, 0),
         datetime.datetime(1994, 11, 5, 9, 0)]

    Example 5: Delta days range, taking hours, minutes and seconds into account

        >>> date = dt.datetime(1994, 11, 5, 7, 23)
        >>> daily_date_range(date, delta_days=-1, reset_time=False, hours=[3, 9])  # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(1994, 11, 4, 9, 23),
         datetime.datetime(1994, 11, 5, 3, 23),
         datetime.datetime(1994, 11, 5, 9, 23)]
    """
    date_today = dt.datetime.today()
    if start_time is None:
        start_time = date_today
    if end_time is None:
        end_time = date_today

    if reset_time:
        # Replace hours, minutesd, seconds and microseconds
        start_time = start_time.replace(
            hour=ref_hour,
            minute=ref_minutes,
            second=ref_seconds,
            microsecond=ref_microseconds,
        )
        end_time = end_time.replace(
            hour=ref_hour,
            minute=ref_minutes,
            second=ref_seconds,
            microsecond=ref_microseconds,
        )

    if delta_days is not None:
        delta_date = time_offset(start_time, days=delta_days)
        if delta_days < 0:
            end_time = start_time
            start_time = delta_date
        else:
            end_time = delta_date

    # If hours is given, set the hour of `end_time` to be the maximum hour
    if hours is not None:
        if isinstance(hours, int):
            max_hour = hours
        else:
            max_hour = max(hours)
        end_time = end_time.replace(hour=max_hour)

    return list(rrule.rrule(freq=rrule.DAILY, dtstart=start_time, until=end_time, byhour=hours))


def monthly_date_range(
    start_time: dt.datetime = None,
    end_time: dt.datetime = None,
    delta_months: int = None,
    reset_time: bool = True,
) -> T.List[dt.datetime]:
    """
    Generate a list of dates with a frequency of 1 month. If `start_time` is `None` today's date is used.
    If `delta_months` is given, a range of dates are generated using this monthly offset and the `start_time`.
    Otherwise, the range is between `start_time` and `end_time`. If `reset_time==True`, the days and time are
    ignored and only the year and month are used (with the days set to 1). This is useful if you want to include the
    first and last month even if they are not complete.

    Args:
        start_time: Start date for range
        end_time: End date for range
        delta_months: The number of calendar months to offset the `start_time`, where the
        generated date range is from this offset date to the `start_time`. Can be positive or negative.
        reset_days: Whether to ignore the days of the datetime, and reset the days to 1

    Returns:
        List of dates

    Example 1: Generate a range of dates with monthly frequency, ignoring the days (reset to 1)

        >>> start_time = dt.datetime(1994, 11, 5)
        >>> end_time = dt.datetime(1995, 3, 1)
        >>> monthly_date_range(start_time, end_time)  # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(1994, 11, 1, 0, 0),
         datetime.datetime(1994, 12, 1, 0, 0),
         datetime.datetime(1995, 1, 1, 0, 0),
         datetime.datetime(1995, 2, 1, 0, 0),
         datetime.datetime(1995, 3, 1, 0, 0)]

    Example 2: Generate a range of dates with monthly frequency, where the day of the start date is retained

        >>> start_time = dt.datetime(1994, 11, 5)
        >>> end_time = dt.datetime(1995, 3, 1)
        >>> monthly_date_range(start_time, end_time, reset_time=False)  # doctest: +NORMALIZE_WHITESPACE
        [datetime.datetime(1994, 11, 5, 0, 0),
         datetime.datetime(1994, 12, 5, 0, 0),
         datetime.datetime(1995, 1, 5, 0, 0),
         datetime.datetime(1995, 2, 5, 0, 0)]

    Example 3: Use a delta months option to generate a range of dates with monthly frequency, starting from
    delta months before or after the start date
        >>> monthly_date_range(dt.datetime(1994, 11, 5), delta_months=1)
        [datetime.datetime(1994, 11, 1, 0, 0), datetime.datetime(1994, 12, 1, 0, 0)]

    """
    date_today = dt.datetime.today()
    if start_time is None:
        start_time = date_today
    if end_time is None:
        end_time = date_today

    if delta_months is not None:
        delta_date = time_offset(start_time, months=delta_months)
        if delta_months < 0:
            end_time = start_time
            start_time = delta_date
        else:
            end_time = delta_date

    if reset_time:
        start_time = dt.datetime(start_time.year, start_time.month, 1)
        end_time = dt.datetime(end_time.year, end_time.month, 1)

    return list(rrule.rrule(freq=rrule.MONTHLY, dtstart=start_time, until=end_time))


def add_day_of_year_variable(
    data: T.Union[xr.Dataset, xr.DataArray] = None, time_dim: str = "time", modify_ordinal_days: bool = True
) -> xr.Dataset:
    """
    Assign a day of year variable to an xr.dataset/dataarray that optionally
    accounts for years with 366 days by repeating day 60 (for February 29).
    Converts xr.DataArray to xr.Dataset so variable can be added.

    Args:
        data: Input dataset/dataarray
        time_dim: Name of time dimension
        modify_ordinal_days: If `True`, then if a year has 366 days, both 29 Feb
        and 1 Mar are assigned day of year 60.

    Returns:
        Dataset with additional dayofyear variable
    """
    day_of_year = get_day_of_year(data, time_dim, modify_ordinal_days=modify_ordinal_days)

    if isinstance(data, xr.DataArray):
        data = data.to_dataset()

    return data.assign({"dayofyear": (time_dim, day_of_year)})


def check_start_end_time_validity(
    start_time: T.Union[dt.datetime, np.datetime64, pd.Timestamp] = None,
    end_time: T.Union[dt.datetime, np.datetime64, pd.Timestamp] = None,
    verbose: bool = False,
) -> bool:
    """
    Check whether end date is after start date

    Args:
        start_time: Start date.
        end_time: End date.
        verbose: Whether to print a warning if end is before start.

    Returns:
        True if end date is after start date, else False

    Example:
        >>> check_start_end_time_validity(dt.datetime(2019, 2, 3), dt.datetime(2012, 1, 3), verbose=False)
        False

    """
    if start_time and end_time:
        if end_time < start_time:
            if verbose:
                print(f"Warning: End time {end_time} before start time {start_time}")
            return False
        else:
            return True
    else:
        raise ValueError("Both `start_time` and `end_time` must be given")


def list_timezones():
    """
    List all available timezones.
    """
    return zoneinfo.available_timezones()


def time_to_local_time(
    times: T.Union[dt.datetime, pd.Timestamp, pd.DatetimeIndex] = None, timezone_name: str = "UTC"
) -> T.Union[dt.datetime, pd.Timestamp, pd.DatetimeIndex]:
    """
    Apply a timezone / daylight-savings-time (dst) offset to a (naive) datetime object.
    The datetimes in `times` are assumed to be in UTC if there are timezone-naive
    (no `tzinfo` has been set). If a timezone is set, the appropriate offset is applied.

    Args:
        times: Datetimes where timezone-naive times are assumed to be in UTC, or the timezone is set.
        timezone_name: The name of the timezone, understood by `pytz.timezone`.
        Use `time.list_timezones()` to view all possible options.

    Returns:
        Datetimes where the required offset has been applied to the time

    Example 1: Central Europe, day before daylight savings time (DST) change. Naive time in UTC.
        >>> date = dt.datetime(2020, 3, 28, 1, 15)
        >>> time_to_local_time(date, timezone_name='CET')
        Timestamp('2020-03-28 02:15:00')

    Example 2: Central Europe, day of daylight savings time (DST) change. Naive time in UTC.
        >>> date = dt.datetime(2020, 3, 29, 1, 15)
        >>> time_to_local_time(date, timezone_name='cet')
        Timestamp('2020-03-29 03:15:00')

    Example 3: Timezone has been set. Convert Egypt time to Central Europe time
        >>> date = dt.datetime(2020, 3, 2, 1, 15, tzinfo=zoneinfo.ZoneInfo('Africa/Cairo'))
        >>> time_to_local_time(date, timezone_name='CET')
        Timestamp('2020-03-02 00:15:00')

    Example 4: Pandas DatetimeIndex. Convert UTC to Sao Paulo, Brazil time
        >>> times = pd.date_range("2001-02-03 ", periods=3, freq='h') # Starts at 2001-02-03T00h00
        >>> time_to_local_time(times, timezone_name='America/Sao_Paulo')
        DatetimeIndex(['2001-02-02 22:00:00', '2001-02-02 23:00:00',
                       '2001-02-03 00:00:00'],
                      dtype='datetime64[ns]', freq=None)
    """
    if not isinstance(timezone_name, str):
        raise TypeError("`timezone_name` must be a `str`")

    if isinstance(times, dt.datetime):
        return _datetimeindex_to_local_time_tz_naive(pd.DatetimeIndex([times]), timezone_name)[0]
    else:
        return _datetimeindex_to_local_time_tz_naive(times, timezone_name)


def _ensure_datetimeindex(
    times: T.Union[dt.datetime, T.Iterable[dt.datetime], pd.DatetimeIndex] = None
) -> pd.DatetimeIndex:
    """
    Ensure a dt.datetime, iterable of dt.datetime, or pd.DateTimeIndex is returned as a pd.DateTimeIndex
    """
    if isinstance(times, pd.DatetimeIndex):
        return times
    elif isinstance(times, dt.datetime):
        return pd.DatetimeIndex([times], name="time")
    else:
        return pd.DatetimeIndex(times, name="time")


def _datetime_to_UTC(times: T.Union[dt.datetime, T.Iterable[dt.datetime], pd.DatetimeIndex] = None) -> pd.DatetimeIndex:
    """
    Ensure a datetime is timezone aware, set to UTC. `times` can be timezone-naive (where no `tz` has been set)
    or have a timezone set.

    Args:
        times: Datetimes - can be timezone naive or have timezone set to 'UTC'

    Returns:
        pd.DateTimeIndex that is timezone aware, set to UTC
    """
    times = _ensure_datetimeindex(times)

    if times.tzinfo is None:
        # Assume times are in UTC and localize time to UTC
        return times.tz_localize(tz="UTC")
    else:
        # Convert times to UTC
        return times.tz_convert("UTC")


def _datetimeindex_to_local_time_tz_aware(
    times: T.Union[dt.datetime, T.Iterable[dt.datetime], pd.DatetimeIndex] = None, timezone_name: str = None
) -> pd.DatetimeIndex:
    """
    Apply a timezone / daylight-savings-time (dst) offset to a datetime-like object. The Timestamps can be
    timezone-naive (no `tz` has been set), where the times are assumed to be in UTC, or have a timezone set.

    Args:
        times: Datetimes can be timezone naive or have timezone set.
        timezone_name: Name of the timezone - understood by `pytz.timezone` or `dateutil.tz.tzfile`

    Returns:
        Datetimes where the required offset has been applied to the time
    """
    # Ensure timezone is set to UTC
    times = _datetime_to_UTC(times)

    if timezone_name is None:
        # Times are assumed to be in UTC
        return times
    else:
        # Apply conversion from UTC to new timezone
        return times.tz_convert(timezone_name)


def _datetimeindex_to_local_time_tz_naive(
    times: T.Union[dt.datetime, T.Iterable[dt.datetime], pd.DatetimeIndex] = None, timezone_name: str = None
) -> pd.DatetimeIndex:
    """
    Apply a timezone / daylight-savings-time (dst) offset to a datetime-like object. The Timestamps are assumed
    to be in UTC - they can be timezone-naive (no `tz` has been set) or have a timezone set.

    Args:
        times: Datetimes where the time is in UTC - can be timezone naive or have timezone set to 'UTC'
        timezone_name: Name of the timezone - understood by `pytz.timezone` or `dateutil.tz.tzfile`

    Returns:
        Datetimes where the required offset has been applied to the time, and the datetimes are returned
        as timezone-naive.
    """
    local_times = _datetimeindex_to_local_time_tz_aware(times, timezone_name)

    # Make times timezone-naive
    local_times = local_times.tz_localize(None)

    return local_times


def utc_offset_in_hours(
    times: T.Union[dt.datetime, T.Iterable[dt.datetime], pd.DatetimeIndex] = None,
    timezone_name: str = None,
    return_single_value: bool = True,
) -> T.Union[float, T.List[float]]:
    """
    Return the offset in (decimal) hours between UTC time and a local timezone, for a given datetime.
    Assumes all datetimes in `times` have the same timezone.

    Args:
        times: Timestamps, assumed to be in UTC (if timezone-naive or no timezone is set). Timestamps can
        be timezone-naive (no `tz` has been set), or have a timezone set.
        timezone_name: Name of the timezone - understood by `pytz.timezone` or `dateutil.tz.tzfile`

    Returns:
        Offset in decimal hours between the given timezone and UTC.
    """
    times = _datetimeindex_to_local_time_tz_aware(times, timezone_name)
    if len(times) == 1 or return_single_value:
        return times[0].utcoffset().total_seconds() / 3600
    else:
        return [time.utcoffset().total_seconds() / 3600 for time in times]


def _set_time_in_data(
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series] = None,
    time_dim: str = "time",
    new_times: pd.DatetimeIndex = None,
    set_time_to_midnight: bool = False,
    hours_to_subtract: float = None,
) -> T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series]:
    """
    Change the timestamps in the `time_dim` of `data`, optionally resetting the time to midnight (00:00:00),
    or subtracting some number of hours (`hours_to_subtract`), or replacing with a new pd.DatatimeIndex.

    Args:
        data: Input data.
        time_dim: Name of the time dimension/index/column in `data` that will be modified.
        new_times: Optional `pd.DatetimeIndex` that will replace the timestamps in `time_dim`. Takes precedence
        over all arguments.
        set_time_to_midnight: If `True`, reset the time part of the timestamps to midnight (00:00:00) on the same day.
        If `False`, do not modify the time part of the timestamps.
        hours_to_subtract: Optional number of hours that will be subtracted from the timestamps in `time_dim`. Takes
        precedence over `set_time_to_midnight`.

    Returns:
        The original `data` with modified timestamps in `time_dim`.
    """
    if new_times is None:  # If no replacement times are given

        if hours_to_subtract is not None:
            # Subtract some number of hours from the times, ignoring `set_time_to_midnight`
            new_times = get_time_from_data(data) - dt.timedelta(hours=hours_to_subtract)

        elif set_time_to_midnight is not True:
            # Nothing to do so return un-modified data
            return data

        else:
            # Reinitialize the time component of the datetime to midnight i.e. 00:00:00, ignoring `hours_to_subtract`
            new_times = get_time_from_data(data).normalize()

    return utils.set_dim_values_in_data(data=data, values=new_times, dim=time_dim)


def data_to_local_time(
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, np.ndarray, T.List, dt.datetime, pd.DatetimeIndex],
    timezone_name: str = None,
    time_dim: str = "time",
):
    """
    Convert and replace times in data to the correct local time, for a given country.
    By default returns original (unmodified) data.

    Args:
        data: Data with time coordinate/index/column. Datetimes are assumed to be in UTC if the timezone is not set.
        timezone_name: The name of the target timezone, understood by `pytz.timezone`.
        Use `list_timezones()` to view all possible options.
        time_dim: The name of the time dimension/coordinate/column.

    Returns:
        Data with modified times

    Example 1: xarray.DataArray
        >>> da = xr.DataArray([1, 2], coords=[pd.date_range("2001-02-03 ", periods=2, freq='h')], dims="time")
        >>> da = data_to_local_time(da, 'America/Sao_Paulo') # Change to Sao Paulo, Brazil local time
        >>> da.indexes['time']
        DatetimeIndex(['2001-02-02 22:00:00', '2001-02-02 23:00:00'], dtype='datetime64[ns]', name='time', freq=None)

    Example 2: pandas.DataFrame
        >>> df = pd.DataFrame(
        ... {"time": pd.date_range("2019-03-15", freq="1D", periods=2), "val": [1, 2]}).set_index(["time"])
        >>> df = data_to_local_time(df, 'America/Sao_Paulo')
        >>> df.index.get_level_values('time')
        DatetimeIndex(['2019-03-14 21:00:00', '2019-03-15 21:00:00'], dtype='datetime64[ns]', name='time', freq=None)
    """
    if timezone_name is None:
        raise ValueError("`timezone_name` must be given")
    else:
        times = get_time_from_data(data)
        times = time_to_local_time(times, timezone_name)

        if isinstance(data, (pd.Series, pd.DataFrame)):
            if time_dim in data.index.names:
                if isinstance(data.index, pd.MultiIndex):
                    data.index = data.index.set_levels(times, level=time_dim)
                else:
                    data.index = times
            else:
                if isinstance(data, pd.Series):
                    # Update values of series, not index
                    data.update(times)
                else:
                    # update column of dataframe
                    data[time_dim] = times
        elif isinstance(data, (xr.Dataset, xr.DataArray)):
            data = data.assign_coords({time_dim: times})
        elif isinstance(data, (np.ndarray, list, dt.datetime, pd.DatetimeIndex)):
            data = times
        else:
            raise TypeError(f"Data type of {type(data)} not implemented")
    return data


################################################

# # Dictionary to define the relative order of pandas time frequencies, where a lower value is higher frequency.
# FREQUENCY_DICT = {
#     "N": 0,
#     "U": 1,
#     "us": 1,
#     "L": 2,
#     "ms": 2,
#     "S": 3,
#     "T": 4,
#     "min": 4,
#     "H": 5,
#     "D": 6,
#     "W": 7,
#     "M": 8,
#     "Q": 9,
#     "A": 10,
#     "Y": 10,
# }


# def parse_date(date: T.Union[str, dt.datetime, pd.Timestamp] = None):
#     """
#     Parse a date string and return a dt.datetime.
#     By default return today's date at time 00:00
#     """
#     if date is None:
#         return dt.datetime.combine(dt.datetime.now(), dt.time.min)
#     elif isinstance(date, str):
#         return dateutil.parser.parse(date, fuzzy=True)
#     else:
#         return date


# def yearly_date_range(
#     start_time: dt.datetime = None,
#     end_time: dt.datetime = None,
#     reset_time: bool = True,
# ) -> T.List[dt.datetime]:
#     """
#     Generate a list of dates with a frequency of 1 year. If `reset_times==True`, the month, day
#     and time are ignored (only using the year values). This is useful if you want to include the
#     first and last year even if they are not complete.

#     Args:
#         start_time: Start date for range
#         end_time: End date for range
#         reset_time: Whether to ignore the month, day and time values of the dates

#     Returns:
#         List of dates

#     Example 1: Generate a range of dates with yearly frequency, ignoring months, days and time

#         >>> start_time = dt.datetime(1994, 11, 5, 7, 23)
#         >>> end_time = dt.datetime(1997, 2, 7, 0)
#         >>> yearly_date_range(start_time, end_time)  # doctest: +NORMALIZE_WHITESPACE
#         [datetime.datetime(1994, 1, 1, 0, 0),
#          datetime.datetime(1995, 1, 1, 0, 0),
#          datetime.datetime(1996, 1, 1, 0, 0),
#          datetime.datetime(1997, 1, 1, 0, 0)]

#     Example 2: Generate a range of dates with yearly frequency, including months, days and time

#         >>> start_time = dt.datetime(1994, 11, 5, 7, 23)
#         >>> end_time = dt.datetime(1997, 2, 7, 0)
#         >>> yearly_date_range(start_time, end_time, reset_time=False)  # doctest: +NORMALIZE_WHITESPACE
#         [datetime.datetime(1994, 11, 5, 7, 23),
#          datetime.datetime(1995, 11, 5, 7, 23),
#          datetime.datetime(1996, 11, 5, 7, 23)]
#     """
#     if reset_time:
#         # Only keep year values
#         start_time = dt.datetime(start_time.year, 1, 1)
#         end_time = dt.datetime(end_time.year, 1, 1)
#     return list(rrule.rrule(freq=rrule.YEARLY, dtstart=start_time, until=end_time))


# def select_period(data, start_time=None, end_time=None):
#     """
#     Selects the period for pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset
#     """
#     start_time = to_datetime64(start_time)
#     end_time = to_datetime64(end_time)
#     check_wf_standard(data)
#     if isinstance(data, (xr.DataArray, xr.Dataset)):
#         return data.sel(time=slice(start_time, end_time))
#     elif isinstance(data, pd.DataFrame):
#         idx = pd.IndexSlice
#         return data.loc[idx[start_time:end_time, :], :]
#     elif isinstance(data, pd.Series):
#         idx = pd.IndexSlice
#         return data.loc[idx[start_time:end_time, :]]


# def _check_valid_resample_method(
#     resample_method: str = None, options: T.List[str] = ["sum", "mean", "min", "max", "cumsum"]
# ):
#     """
#     Check if the resample method string is valid
#     """
#     if resample_method in options:
#         return True
#     else:
#         raise ValueError(f"`resample_method` = '{resample_method}' is not a valid option")


# def resample_time(
#     data,
#     freq="D",
#     resample_method="sum",
#     day_start_hour: float = 0,
#     timezone_name: str = None,
#     return_local_time: bool = False,
#     set_time_to_midnight: bool = False,
#     data_timestep_freq: float = 6,
#     time_dim: str = "time",
#     keep_attrs=True,
#     **kwargs,
# ):
#     """
#     Resample data in time, taking account of the timezone. The data can be timezone-naive (assumed to be UTC),
#     or have a timezone set.

#     Args:
#         data: Data to resample
#         freq: Resample time frequency based on pd/xr resample method, options ('D', 'W', ...).
#         resample_method: Resample method ('mean', 'sum', 'max' or 'min').
#         day_start_hour: The hour of the day that is used as the start of the day (so 1 complete day is from
#         `day_start_hour` to `day_start_hour` + 24h).
#         timezone_name: Name of the timezone, understood by `pytz.timezone` or `dateutil.tz.tzfile`
#         return_local_time: Return the data with the local times instead of the resample times, where the
#         resample times are taken as the nearest multiple of the `data_timestep_freq` for a given UTC
#         offset offset (i.e. for 6h timesteps and timezone UTC-5, the data are resampled using multiples of 6h,
#         where a UTC timestamp of 12:00 becomes 06:00). Resample times may be different from local time by up
#         to 3h for 6h timesteps (in general, different by `n/2`h for `n`h timesteps).
#         set_time_to_midnight: If `True`, reset the time part of the timestamps to midnight (00:00:00) on the same day.
#         If `False`, do not modify the time part of the timestamps.
#         data_timestep_freq: Data frequency in hours. Only used if < 24. Required if `return_local_time==True`.
#         time_dim: Name of the time dimension/index/column in `data` that will be modified.
#         keep_attrs (bool, optional): In case of xr keep attributes, defaults to True.

#     Returns:
#         Data resampled in time
#     """
#     check_wf_standard(data)

#     if timezone_name is None:
#         # Assume timezone of data is UTC, so both the resample offset and the UTC offset are zero
#         utc_resample_offset = 0
#         utc_offset_hours = 0
#     else:
#         if data_timestep_freq is None:
#             raise ValueError("`data_timestep_freq` must be given when resampling in a non-UTC timezone.")

#         times = get_time_from_data(data)

#         # Get appropriate UTC offset to apply to the data
#         utc_offset_hours = utc_offset_in_hours(times, timezone_name)

#         if data_timestep_freq < 24:
#             utc_resample_offset = -1 * wfutils.nearest_multiple(utc_offset_hours, base=data_timestep_freq)
#         else:
#             # If `data_timestep_freq more than 24h, the resample offset should be zero, as resampling
#             # ignores the hours
#             utc_resample_offset = 0.0

#     # The UTC offset in hours to apply to the data, accounting for both the timezone and the start hour of the day
#     resample_offset = dt.timedelta(hours=utc_resample_offset + day_start_hour)

#     if isinstance(data, (xr.DataArray, xr.Dataset)):

#         # recast `offset` to an `int` since xarray expects an `int` dtype for the `base` argument.
#         # This could be removed with a future release of xarray, as long as the `base` argument is replaced by
#         # the same `offset` argument as pandas resample
#         resample_offset = int(resample_offset.total_seconds() / 3600)

#         resampled = _resample_time_xr(
#             ds=data,
#             freq=freq,
#             resample_method=resample_method,
#             base=resample_offset,
#             loffset=dt.timedelta(hours=resample_offset),
#             keep_attrs=keep_attrs,
#             **kwargs,
#         )
#     elif isinstance(data, (pd.DataFrame, pd.Series)):
#         resampled = _resample_time_pd(
#             df=data, freq=freq, resample_method=resample_method, offset=resample_offset, **kwargs
#         )

#     if freq is not None:
#         _, char = split_freq(freq)

#         # For frequencies of 1 day or higher
#         if char in ("H", "T", "min", "S", "L", "ms", "U", "us", "N"):

#             # Never reset time values to midnight as this leads to non-unique indexes
#             set_time_to_midnight = False

#     # If local times are required, subtract the UTC offset from the timestamps
#     if return_local_time:
#         hours_to_subtract = -1 * utc_offset_hours
#     else:
#         hours_to_subtract = None

#     return _set_time_in_data(
#         data=resampled,
#         time_dim=time_dim,
#         set_time_to_midnight=set_time_to_midnight,
#         hours_to_subtract=hours_to_subtract,
#     )

#     return resampled


# def _resample_time_xr(ds, freq="D", resample_method="sum", keep_attrs=True, base=None, **kwargs):
#     """ Resample xr.DataArray/xr.Dataset in time """
#     if freq is None:
#         return ds
#     elif freq == "all":
#         if resample_method == "sum":
#             return ds.sum(dim="time", keep_attrs=keep_attrs)
#         elif resample_method == "mean":
#             return ds.mean(dim="time", keep_attrs=keep_attrs)
#         elif resample_method == "min":
#             return ds.min(dim="time", keep_attrs=keep_attrs)
#         elif resample_method == "max":
#             return ds.max(dim="time", keep_attrs=keep_attrs)
#         elif resample_method == "cumsum":
#             return ds.cumsum(dim="time", keep_attrs=keep_attrs)
#     else:
#         ds_resampled = ds.resample(time=freq, base=base, **kwargs)
#         if resample_method == "sum":
#             return ds_resampled.sum(keep_attrs=keep_attrs)
#         elif resample_method == "mean":
#             return ds_resampled.mean(keep_attrs=keep_attrs)
#         elif resample_method == "min":
#             return ds_resampled.min(keep_attrs=keep_attrs)
#         elif resample_method == "max":
#             return ds_resampled.max(keep_attrs=keep_attrs)
#         elif resample_method == "cumsum":
#             grouped_ds = _groupby_freq_xr(ds=ds, freq=freq)
#             ds_cumsum = grouped_ds.map(lambda x: x.cumsum(dim="time", skipna=True, keep_attrs=keep_attrs))
#             return ds_cumsum.reindex(time=ds.indexes["time"])
#             # Only valid for xr.DataArray
#             # ds_cumsum = ds_resampled.map(lambda x: np.cumsum(x))
#             # return ds_cumsum.assign_coords({'time': ds.indexes["time"]})


# def _resample_time_pd(df, freq="D", resample_method="sum", offset=None, **kwargs):
#     """ Resample pd.Series/pd.DataFrame for the standar weatherforce MultiIndex (['time', 'id']) format """

#     if freq is not None:
#         if freq == "all":
#             grouped_df = df.groupby(level=["id"])
#         else:
#             grouped_df = df.groupby(
#                 [pd.Grouper(level="time", freq=freq, offset=offset, **kwargs), pd.Grouper(level="id")]
#             )

#         if resample_method == "sum":
#             resampled_df = grouped_df.sum()
#         elif resample_method == "mean":
#             resampled_df = grouped_df.mean()
#         elif resample_method == "min":
#             resampled_df = grouped_df.min()
#         elif resample_method == "max":
#             resampled_df = grouped_df.max()
#         elif resample_method == "cumsum":
#             resampled_df = grouped_df.cumsum()
#         return resampled_df
#     else:
#         return df


# def groupby_freq(data, freq, **kwargs):
#     """
#     Groupby pandas freq

#     Args:
#         data (xr.DataArray, xr.Dataset, pd.Series, pd.DataFrame): Data to groupby
#         freq (str): Groupby time frequency based on pd/xr resample method
#         options ('D', 'W', ...).
#     Returns:
#         (xr.DataArrayGroupBy, xr.DatasetGroupBy, pd.SeriesGroupBy, pd.DataFrameGroupBy): Data grouped in time.
#     """
#     check_wf_standard(data)
#     if isinstance(data, (xr.DataArray, xr.Dataset)):
#         return _groupby_freq_xr(ds=data, freq=freq, **kwargs)
#     elif isinstance(data, (pd.DataFrame, pd.Series)):
#         return data.groupby([pd.Grouper(level="time", freq=freq), pd.Grouper(level="id")])


# def _groupby_freq_xr(ds, freq, **kwargs):
#     """
#     Returns a groupby object for the given freq for a xr.Dataset/xr.DataArray

#     Example 1:  Groupby month
#         >>> ds = xr.Dataset({"tp": (("lat", "lon", "time"),
#         ...                         np.ones((11, 11, 365)), {"name": "test_dataset"})},
#         ...                 coords={'lat': np.arange(-5, 5 + 1),
#         ...                         'lon': np.arange(-5, 5 + 1),
#         ...                         "time": pd.date_range("2019-01-01", periods=365)},
#         ...                 attrs={"name": "test_dataset"})
#         >>> _groupby_freq_xr(ds, "M")
#         DatasetGroupBy, grouped over 'time'
#         12 groups with labels 2019-01-31, ..., 2019-12-31.

#     Example 2: Groupby year
#         >>> ds = xr.Dataset({"tp": (("lat", "lon", "time"),
#         ...                         np.ones((11, 11, 365)), {"name": "test_dataset"})},
#         ...                 coords={'lat': np.arange(-5, 5 + 1),
#         ...                         'lon': np.arange(-5, 5 + 1),
#         ...                         "time": pd.date_range("2019-01-01", periods=365)},
#         ...                 attrs={"name": "test_dataset"})
#         >>> _groupby_freq_xr(ds, "Y")
#         DatasetGroupBy, grouped over 'time'
#         1 groups with labels 2019-12-31.

#     Example 3: Groupby ndays
#         >>> ds = xr.Dataset({"tp": (("lat", "lon", "time"),
#         ...                         np.ones((11, 11, 365)), {"name": "test_dataset"})},
#         ...                 coords={'lat': np.arange(-5, 5 + 1),
#         ...                         'lon': np.arange(-5, 5 + 1),
#         ...                         "time": pd.date_range("2019-01-01", periods=365)},
#         ...                 attrs={"name": "test_dataset"})
#         >>> _groupby_freq_xr(ds, "7D")
#         DatasetGroupBy, grouped over 'time'
#         53 groups with labels 2019-01-01, ..., 2019-12-31.
#     """

#     time_index = ds.indexes["time"]
#     original_time = pd.DataFrame({"time": time_index}).set_index("time")
#     grouped_time = original_time.groupby(pd.Grouper(level="time", freq=freq))  # .ngroup().reset_index()
#     labels = []
#     prev_item = 0
#     for key, item in grouped_time.groups.items():
#         labels += [key] * (item - prev_item)
#         prev_item = item
#     labels = xr.DataArray(labels, coords={"time": time_index}, dims="time", name="time")
#     return ds.groupby(labels, **kwargs)


# # def freq_to_timedelta64(freq):
# #     """
# #     Transforms pd.resample method freq to timedelta64 in ns

# #     Args:
# #         freq (str): Resample freq (s, m, H, D, Y) or a multiple of them (for example "4D")
# #     Returns:
# #         (np.timedelta64) A timedelta64 in ns

# #     example 1 : Day
# #     >>> freq_to_timedelta64("D")
# #     numpy.timedelta64(86400000000000,'ns')

# #     example 2 : iHour
# #     >>> freq_to_timedelta64("12H")
# #     numpy.timedelta64(43200000000000,'ns')

# #     example 3 : ihour
# #     >>> freq_to_timedelta64("6h")
# #     numpy.timedelta64(21600000000000,'ns')

# #     example 4 : Month
# #     >>> freq_to_timedelta64("M") is None
# #     True
# #     """
# #     step, resample_freq = split_freq(freq)

# #     if resample_freq is not None:
# #         if resample_freq == "M":
# #             logging.warning("M leads to ambiguous timedelta64. Unable to determine the timedelta64.")
# #         elif len(resample_freq) == 1 and 0 < len(step):
# #             if resample_freq == "H":
# #                 resample_freq = resample_freq.lower()
# #             return np.timedelta64(int(step), resample_freq).astype("timedelta64[ns]")
# #         elif len(resample_freq) == 1 and len(step) == 0:
# #             if resample_freq == "H":
# #                 resample_freq = resample_freq.lower()
# #             return np.timedelta64(1, resample_freq).astype("timedelta64[ns]")
# #     else:
# #         raise ValueError("Only s, m, h, H, D, Y are accepted as freq")


# # def split_freq(freq):
# #     """
# #     Splits the resample freq into the step size and the time freq

# #     example 1 : Day
# #     >>> split_freq("D")
# #     ('', 'D')

# #     example 2 : iHour
# #     >>> split_freq("6H")
# #     ('6', 'H')

# #     example 3 : iMonthStart
# #     >>> split_freq("5MS")
# #     ('5', 'M')

# #     example 4 : Other
# #     >>> split_freq("BS")
# #     ('', None)
# #     """
# #     step = ""
# #     resample_freq = None
# #     for char in freq:
# #         if char.isdigit():
# #             step += char
# #         elif char in "smhHDMY":
# #             resample_freq = char
# #             break
# #     return step, resample_freq
