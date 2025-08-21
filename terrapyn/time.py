import datetime as dt
import typing as T
import zoneinfo

import numpy as np
import pandas as pd
import xarray as xr
from dateutil import rrule
from dateutil.relativedelta import relativedelta

import terrapyn as tp

# import dateutil


def datetime64_to_datetime(datetime64: np.datetime64 = None) -> dt.datetime:
	"""Convert numpy.datetime64 to dt.datetime"""
	return np.datetime64(datetime64, "us").astype(dt.datetime)


def datetime_to_datetime64(time: dt.datetime = None) -> np.datetime64:
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
	data: xr.Dataset
	| xr.DataArray
	| pd.DataFrame
	| pd.Series
	| np.ndarray
	| list
	| dt.datetime
	| pd.DatetimeIndex
	| pd.MultiIndex = None,
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
	if isinstance(data, pd.Series | pd.DataFrame):
		# Check if `time_dim` is the index
		if time_dim in data.index.names:
			times = data.index.get_level_values(time_dim)
		else:
			times = pd.DatetimeIndex(data) if isinstance(data, pd.Series) else pd.DatetimeIndex(data[time_dim])
	elif isinstance(data, xr.Dataset | xr.DataArray):
		times = data.indexes[time_dim]
	elif isinstance(data, np.ndarray | list):
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
				list,
				dt.datetime,
				pd.DatetimeIndex,
			]
		)
		raise TypeError(f"Data is of type {type(data)} but must be one of type: {data_types_str}")
	return times


def groupby_time(
	data: xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series = None,
	time_dim: str = "time",
	grouping: str = "week",
	other_grouping_keys: str | list[str] | None = None,
):
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
	times = get_time_from_data(data, time_dim=time_dim)

	if grouping == "week":
		time_groups = times.isocalendar().week
	elif grouping == "month":
		time_groups = times.month.to_numpy()
	elif grouping == "year":
		time_groups = times.year.to_numpy()
	elif grouping == "dayofyear":
		time_groups = get_day_of_year(times, time_dim, modify_ordinal_days=True)
	elif grouping == "dekad":
		time_groups = datetime_to_dekad_number(times)
	elif grouping == "pentad":
		time_groups = datetime_to_pentad_number(times)
	else:
		raise ValueError("`grouping` must be one of 'dayofyear', 'week', 'month', 'year', 'pentad', 'dekad'")

	if isinstance(time_groups, pd.Series):
		time_groups = time_groups.to_numpy()

	if isinstance(data, xr.DataArray | xr.Dataset):
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
	data: (
		xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series | np.ndarray | list | dt.datetime | pd.DatetimeIndex
	) = None,
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
		ordinal_day = times.dayofyear.to_numpy()
		modified_ordinal_day = ordinal_day
		modified_ordinal_day[march_or_later] -= 1
		return modified_ordinal_day
	else:
		return times.dayofyear.to_numpy()


def datetime_to_pentad_number(
	times: pd.DatetimeIndex | pd.Series | dt.datetime | list | np.ndarray = None,
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
	dates: pd.DatetimeIndex | pd.Series | dt.datetime | np.datetime64 | list | np.ndarray = None,
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
		>>> datetime_to_dekad_number(np.array([np.datetime64("2004-01-01"), np.datetime64("2004-02-05")]))
		array([1, 4])
		>>> datetime_to_dekad_number([dt.datetime(2004, 1, 1), dt.datetime(2004, 1, 11), dt.datetime(2004, 2, 5)])
		array([1, 2, 4])
		>>> datetime_to_dekad_number(dt.datetime(2004, 2, 5))
		array([4])

	"""
	if isinstance(dates, dt.datetime | np.datetime64):
		dates = [dates]
	dates = pd.DatetimeIndex(dates)
	count = np.digitize(dates.day, bins=[10, 20], right=True) + 1  # Add 1 since bins start at zero
	dekads = (dates.month - 1) * 3 + count
	return dekads.to_numpy()


def daily_date_range(
	start_time: dt.datetime = None,
	end_time: dt.datetime = None,
	delta_days: int = None,
	hours: int | list[int] = None,
	reset_time: bool = True,
	ref_hour: int = 0,
	ref_minutes: int = 0,
	ref_seconds: int = 0,
	ref_microseconds: int = 0,
) -> list[dt.datetime]:
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
		max_hour = hours if isinstance(hours, int) else max(hours)
		end_time = end_time.replace(hour=max_hour)

	return list(rrule.rrule(freq=rrule.DAILY, dtstart=start_time, until=end_time, byhour=hours))


def monthly_date_range(
	start_time: dt.datetime = None,
	end_time: dt.datetime = None,
	delta_months: int = None,
	reset_time: bool = True,
) -> list[dt.datetime]:
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
	data: xr.Dataset | xr.DataArray = None, time_dim: str = "time", modify_ordinal_days: bool = True
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
	start_time: dt.datetime | np.datetime64 | pd.Timestamp = None,
	end_time: dt.datetime | np.datetime64 | pd.Timestamp = None,
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
	times: dt.datetime | pd.Timestamp | pd.DatetimeIndex = None, timezone_name: str = "UTC"
) -> dt.datetime | pd.Timestamp | pd.DatetimeIndex:
	"""
	Apply a timezone / daylight-savings-time (dst) offset to a (naive) datetime object.
	The datetimes in `times` are assumed to be in UTC if there are timezone-naive
	(no `tzinfo` has been set). If a timezone is set, the appropriate offset is applied.

	Args:
		times: Datetimes where timezone-naive times are assumed to be in UTC, or the timezone is set.
		timezone_name: The name of the timezone, understood by `zoneinfo`.
		Use `time.list_timezones()` to view all possible options.

	Returns:
		Datetimes where the required offset has been applied to the time

	Example 1: Central Europe, day before daylight savings time (DST) change. Naive time in UTC.
		>>> date = dt.datetime(2020, 3, 28, 1, 15)
		>>> time_to_local_time(date, timezone_name="CET")
		Timestamp('2020-03-28 02:15:00')

	Example 2: Central Europe, day of daylight savings time (DST) change. Naive time in UTC.
		>>> date = dt.datetime(2020, 3, 29, 1, 15)
		>>> time_to_local_time(date, timezone_name="cet")
		Timestamp('2020-03-29 03:15:00')

	Example 3: Timezone has been set. Convert Egypt time to Central Europe time
		>>> date = dt.datetime(2020, 3, 2, 1, 15, tzinfo=zoneinfo.ZoneInfo("Africa/Cairo"))
		>>> time_to_local_time(date, timezone_name="CET")
		Timestamp('2020-03-02 00:15:00')

	Example 4: Pandas DatetimeIndex. Convert UTC to Sao Paulo, Brazil time
		>>> times = pd.date_range("2001-02-03 ", periods=3, freq="h")  # Starts at 2001-02-03T00h00
		>>> time_to_local_time(times, timezone_name="America/Sao_Paulo")  # doctest: +NORMALIZE_WHITESPACE
		DatetimeIndex(['2001-02-02 22:00:00', '2001-02-02 23:00:00',
		       '2001-02-03 00:00:00'],
		      dtype='datetime64[ns]', freq=None)
	"""  # noqa: E101
	if not isinstance(timezone_name, str):
		raise TypeError("`timezone_name` must be a `str`")

	if isinstance(times, dt.datetime):
		return _datetimeindex_to_local_time_tz_naive(pd.DatetimeIndex([times]), timezone_name)[0]
	else:
		return _datetimeindex_to_local_time_tz_naive(times, timezone_name)


def _ensure_datetimeindex(
	times: dt.datetime | T.Iterable[dt.datetime] | pd.DatetimeIndex = None,
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


def _datetime_to_UTC(times: dt.datetime | T.Iterable[dt.datetime] | pd.DatetimeIndex = None) -> pd.DatetimeIndex:
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
	times: dt.datetime | T.Iterable[dt.datetime] | pd.DatetimeIndex = None, timezone_name: str = None
) -> pd.DatetimeIndex:
	"""
	Apply a timezone / daylight-savings-time (dst) offset to a datetime-like object. The Timestamps can be
	timezone-naive (no `tz` has been set), where the times are assumed to be in UTC, or have a timezone set.

	Args:
		times: Datetimes can be timezone naive or have timezone set.
		timezone_name: Name of the timezone, understood by `zoneinfo`

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
	times: dt.datetime | T.Iterable[dt.datetime] | pd.DatetimeIndex = None, timezone_name: str = None
) -> pd.DatetimeIndex:
	"""
	Apply a timezone / daylight-savings-time (dst) offset to a datetime-like object. The Timestamps are assumed
	to be in UTC - they can be timezone-naive (no `tz` has been set) or have a timezone set.

	Args:
		times: Datetimes where the time is in UTC - can be timezone naive or have timezone set to 'UTC'
		timezone_name: Name of the timezone, understood by `zoneinfo`

	Returns:
		Datetimes where the required offset has been applied to the time, and the datetimes are returned
		as timezone-naive.
	"""
	local_times = _datetimeindex_to_local_time_tz_aware(times, timezone_name)

	# Make times timezone-naive
	local_times = local_times.tz_localize(None)

	return local_times


def utc_offset_in_hours(
	times: dt.datetime | T.Iterable[dt.datetime] | pd.DatetimeIndex | pd.Timestamp = None,
	timezone_name: str = None,
	return_single_value: bool = True,
) -> float | list[float]:
	"""
	Return the offset in (decimal) hours between UTC time and a local timezone, for a given datetime.
	Assumes all datetimes in `times` have the same timezone.

	Args:
		times: Timestamps, assumed to be in UTC (if timezone-naive or no timezone is set). Timestamps can
		be timezone-naive (no `tz` has been set), or have a timezone set.
		timezone_name: Name of the timezone, understood by `zoneinfo`

	Returns:
		Offset in decimal hours between the given timezone and UTC.
	"""
	times = _datetimeindex_to_local_time_tz_aware(times, timezone_name)
	if len(times) == 1 or return_single_value:
		return times[0].utcoffset().total_seconds() / 3600
	else:
		return [time.utcoffset().total_seconds() / 3600 for time in times]


def _set_time_in_data(
	data: xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series = None,
	time_dim: str = "time",
	new_times: pd.DatetimeIndex = None,
	set_time_to_midnight: bool = False,
	hours_to_subtract: float = None,
) -> xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series:
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
			new_times = get_time_from_data(data, time_dim=time_dim) - dt.timedelta(hours=hours_to_subtract)

		elif set_time_to_midnight is not True:
			# Nothing to do so return un-modified data
			return data

		else:
			# Reinitialize the time component of the datetime to midnight i.e. 00:00:00, ignoring `hours_to_subtract`
			new_times = get_time_from_data(data, time_dim=time_dim).normalize()

	return tp.utils.set_dim_values_in_data(data=data, values=new_times, dim=time_dim)


def data_to_local_time(
	data: (
		pd.Series
		| pd.DataFrame
		| xr.Dataset
		| xr.DataArray
		| np.ndarray
		| list
		| dt.datetime
		| pd.DatetimeIndex
		| pd.MultiIndex
	) = None,
	timezone_name: str = "UTC",
	time_dim: str = "time",
) -> pd.DatetimeIndex:
	"""
	Convert and replace times in data to the correct local time, for a given country.
	By default returns original (unmodified) data.

	Args:
		data: Data with time coordinate/index/column. Datetimes are assumed to be in UTC if the timezone is not set.
		timezone_name: The name of the target timezone, understood by `zoneinfo`.
		Use `list_timezones()` to view all possible options.
		time_dim: The name of the time dimension/coordinate/column.

	Returns:
		Data with modified times

	Example 1: xarray.DataArray
		>>> da = xr.DataArray([1, 2], coords=[pd.date_range("2001-02-03 ", periods=2, freq="h")], dims="time")
		>>> da = data_to_local_time(da, "America/Sao_Paulo")  # Change to Sao Paulo, Brazil local time
		>>> da.indexes["time"]
		DatetimeIndex(['2001-02-02 22:00:00', '2001-02-02 23:00:00'], dtype='datetime64[ns]', name='time', freq=None)

	Example 2: pandas.DataFrame
		>>> df = pd.DataFrame({"time": pd.date_range("2019-03-15", freq="1D", periods=2),
		...						"val": [1, 2]}).set_index(["time"])
		>>> df = data_to_local_time(df, "America/Sao_Paulo")
		>>> df.index.get_level_values("time")
		DatetimeIndex(['2019-03-14 21:00:00', '2019-03-15 21:00:00'], dtype='datetime64[ns]', name='time', freq=None)
	"""  # noqa
	if timezone_name is None:
		raise ValueError("`timezone_name` must be given")
	else:
		times = get_time_from_data(data, time_dim=time_dim)
		times = time_to_local_time(times, timezone_name)

		if isinstance(data, np.ndarray | list | dt.datetime | pd.DatetimeIndex):
			return times
		else:
			return tp.utils.set_dim_values_in_data(data=data, values=times, dim=time_dim)


def _resample_pandas_multiindex(
	data: pd.DataFrame | pd.Series = None,
	time_dim: str = "time",
	freq: str = "D",
	offset: int | float = None,
	closed: str = "left",
) -> pd.DataFrame | pd.Series:
	"""
	Resample a pd.Dataframe/Series with a pd.Multiindex, where we do not reduce over other index levels,
	only `time_dim`.

	Args:
		data: Data to resample in time
		time_dim: Name of the time dimension/index/column in `data` that will be used to determine the grouping.
		freq: Resample frequency. Follows Pandas notation here
		https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

	Returns:
		Groupby object with data grouped by the given frequency.
	"""
	time_dim_level = data.index.names.index(time_dim)
	other_levels = np.delete(data.index.names, time_dim_level)
	grouped = data.groupby([pd.Grouper(level=time_dim, freq=freq, offset=offset, closed=closed), *other_levels])
	return grouped


def groupby_freq(
	data,
	time_dim: str = "time",
	freq: str = "D",
	closed: str = "left",
	day_start_hour: int = 0,
	other_grouping_keys: list = None,
):
	"""
	Group data by time.

	Args:
		data: Data to resample in time
		time_dim: Name of the time dimension/index/column in `data` that will be used to determine the grouping.
		freq: Resample frequency. Follows Pandas notation here
		https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
		day_start_hour: The hour that is used as the start of the day (so 1 complete day is from
		`day_start_hour` to `day_start_hour` + 24h). Defaults to 0h.

	Returns:
		Groupby object with data grouped by the given frequency.

	Example: xr.Dataset / pd.DataFrame group by month
		>>> ds = xr.Dataset(
		...     data_vars={"var": (("lat", "lon", "time"), np.ones((1, 1, 100)))},
		...     coords={"lat": [1], "lon": [2], "time": pd.date_range("2022-01-01", periods=100)},
		... )
		>>> groupby_freq(ds["var"].to_dataframe(), freq="ME").sum()  # doctest: +NORMALIZE_WHITESPACE
							var
		time       lat lon
		2022-01-31 1   2    30.0
		2022-02-28 1   2    28.0
		2022-03-31 1   2    31.0
		2022-04-30 1   2    11.0
	"""
	# The offset in hours to apply to the data to account for the day start hour
	offset = pd.Timedelta(hours=day_start_hour)

	if isinstance(data, xr.DataArray | xr.Dataset):
		# Bug with xarray.Dataset.resample where it ignores the `base` argument, so a workaround is
		# implemented here, where the times are modified and `loffset` is used to re-label the time
		# return _set_time_in_data(data, time_dim=time_dim, hours_to_subtract=day_start_hour).resample(
		#     {time_dim: freq}, loffset=str(day_start_hour) + "H", closed=closed
		# )
		return data.resample({time_dim: freq}, offset=offset, closed=closed)

	elif isinstance(data, pd.Series | pd.DataFrame):
		is_dim_in_index = tp.utils._dim_in_pandas_index(data.index, time_dim)
		if is_dim_in_index:
			if tp.utils._pandas_check_multiindex_type(data.index):
				# Is multiindex
				return _resample_pandas_multiindex(
					data=data, time_dim=time_dim, freq=freq, offset=offset, closed=closed
				)
			else:
				return data.groupby(pd.Grouper(level=time_dim, freq=freq, offset=offset, closed=closed))
		else:
			if isinstance(data, pd.Series):
				raise ValueError("Cannot group by time in time if data is a pandas.Series")
			elif isinstance(data, pd.DataFrame):
				# time_dim not in index, look in columns for DataFrames
				if time_dim not in data.columns:
					raise ValueError(f"time_dim=`{time_dim}` not found in data")

				if other_grouping_keys:
					other_grouping_keys = tp.utils.ensure_list(other_grouping_keys)
					return data.groupby(
						[pd.Grouper(key=time_dim, freq=freq, offset=offset, closed=closed), *other_grouping_keys]
					)
				else:
					return data.groupby(pd.Grouper(key=time_dim, freq=freq, offset=offset, closed=closed))

	else:
		data_types_str = ", ".join(
			str(i)
			for i in [
				xr.Dataset,
				xr.DataArray,
				pd.DataFrame,
				pd.Series,
			]
		)
		raise TypeError(f"Data is of type {type(data)} but must be one of type: {data_types_str}")


def resample_time(
	data: pd.DataFrame | pd.Series = None,
	time_dim: str = "time",
	freq: str = "D",
	closed: str = "left",
	resample_method: str = "sum",
	day_start_hour: int = 0,
	other_grouping_keys: list = None,
):
	"""
	Resample data in time.

	Args:
		data: Data to resample in time
		time_dim: Name of the time dimension/index/column in `data` that will be used to determine the aggregation.
		freq: Resample frequency. Follows Pandas notation here
		https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
		resample_method: How to reduce/aggregate the data in each resampled group, one
		of 'mean', 'sum', 'max', 'min', 'cumsum'.
		day_start_hour: The hour that is used as the start of the day (so 1 complete day is from
		`day_start_hour` to `day_start_hour` + 24h). Defaults to 0h.

	Returns:
		Data resampled in time
	"""
	grouped = groupby_freq(
		data,
		time_dim=time_dim,
		freq=freq,
		closed=closed,
		day_start_hour=day_start_hour,
		other_grouping_keys=other_grouping_keys,
	)

	# if resample_method == 'cumsum':

	#     time_index = tp.utils.get_dim_values_from_data(data, time_dim)
	#     # For xarray
	#     if isinstance(grouped, (xr.core.groupby.DatasetGroupBy, xr.core.groupby.DataArrayGroupBy)):
	# cumsum = grouped.map(lambda x: x.cumsum(dim=dim, skipna=True, keep_attrs=True))
	# return cumsum.reindex({time_dim: time_index})
	#     elif isinstance(grouped, (pd.core.groupby.generic.DataFrameGroupBy, pd.core.groupby.generic.SeriesGroupBy)):
	# return grouped.cumsum(**kwargs)

	return tp.utils._call_resample_method(grouped, resample_method)


def rolling(
	data: pd.Series | xr.DataArray = None,
	n_periods: int = 3,
	min_periods: int = None,
	method: str = "sum",
	time_dim: str = "time",
) -> pd.Series | xr.DataArray:
	"""
	Calculate a rolling `method` over a window with `n_periods`, with minimum of `min_periods` in the window.
	Works for both pd.DataFrame/Series and xr.DataArray/xr.Dataset

	Args:
		data: pd.Dataframe/Series or xr.Dataset/DataArray.
		n_periods: Number of periods to include in the sliding window, where the given `method` is applied
			to values from `time - n_periods` to `time`. i.e. `n_periods = 3` with `method='sum'` means the
			data at a given time will
		be summed with the previous 2 time steps

	"""
	if min_periods is None:
		min_periods = n_periods

	if isinstance(data, pd.Series):
		rolling_data = data.rolling(n_periods, min_periods=n_periods)
	elif isinstance(data, xr.DataArray):
		rolling_data = data.rolling({time_dim: n_periods}, min_periods=n_periods)
	else:
		raise TypeError("data must be of type pd.Series or xr.DataArray")

	return tp.utils._call_resample_method(rolling_data, method, dim=time_dim)


def disaggregate_to_daily(
	data: pd.DataFrame = None,
	time_dim: str = "time",
	n_days_in_period: int = 8,
	agg_type: str = "sum",
	normalize_year_end: bool = True,
) -> pd.DataFrame:
	"""
	Dissaggregate data with multi-day period (e.g. 8-day, 16-day) to daily values.

	If `agg_type=='sum'` then the values are divided by the number of days in each period, to give
	the mean daily value. If `agg_type=='mean'` then the values are duplicated for each day in the period.

	Optionally takes care of incomplete periods at the end of a calendar year and normalizes data using
	however many days are in that incomplete period. e.g. for 8-day periods where the 'date' is the start
	of the 8 day period, the last period of each year covers 5 or 6 days. For daily values, the data are
	normalized by 8 days apart from the last periods in the year which are normalized by 5 or 6 days.

	Args:
		data: Input data
		n_days_in_period: Number of days in period
		agg_type: Type of aggregation used in creating the input data. Options are 'sum', 'mean'.
		This determines how the values are normalized (divided or duplicated).
		normalize_year_end: If True, normalize the last period of each year by the number of days in the period

	Returns:
		pd.DataFrame with index of 'date' and values resampled to daily values
	"""
	# Extract/Convert the `time_dim` to a pandas.DatetimeIndex
	times = get_time_from_data(data, time_dim=time_dim)

	# Daily date range from period start date to period end date + n_days_in_period
	# unless end date is in last period of year, then end of year
	start = times[0]
	end = times[-1]
	if (end.month == 12) & (31 - end.day < n_days_in_period):
		end = pd.Timestamp(year=end.year, month=12, day=31)
		fill_until_year_end = True
	else:
		end = end + pd.Timedelta(n_days_in_period - 1, unit="D")
		fill_until_year_end = False

	daily = pd.date_range(start, end, freq="D")

	orig_daily = data.reindex(index=daily, method="ffill").copy()

	if agg_type == "sum":
		if normalize_year_end:
			# Divide all values by n_days_in_period apart from last period in year
			# unless fill_until_year_end == False, then just divide last day by n_days_in_period
			factor = pd.Series(data=n_days_in_period, index=times)

			if fill_until_year_end:
				# how many days until end of year for each of these dates
				end_of_year_dates = factor.index[
					(factor.index.month == 12) & (31 - factor.index.day < n_days_in_period)
				]
				end_of_year_factors = [
					(366 - date.day_of_year + 1) if date.is_leap_year else (365 - date.day_of_year + 1)
					for date in end_of_year_dates
				]
				factor[end_of_year_dates] = end_of_year_factors

			factor_daily = factor.reindex(index=daily, method="ffill")
			orig_daily = orig_daily.div(factor_daily, axis=0)
		else:
			# divide all by n_days_in_period
			orig_daily = orig_daily.div(n_days_in_period)

	return orig_daily


def test_disaggregate_to_daily():
	"""
	Test disaggregate_to_daily
	"""
	# Create a dataframe with 8-day period data
	dates = pd.date_range("2022-01-01", freq="8D", periods=2, name="time")
	data = pd.DataFrame({"data": np.full(len(dates), 8)}, index=dates)

	disaggregated = disaggregate_to_daily(data, n_days_in_period=8, agg_type="sum", normalize_year_end=True)

	daily_dates = pd.date_range("2022-01-01", freq="D", periods=16, name="time")
	expected = pd.Series(index=daily_dates, data=np.full(len(daily_dates), 1.0))

	assert disaggregated["data"].equals(expected)


def yearly_date_range(
	start_time: dt.datetime = None,
	end_time: dt.datetime = None,
	reset_time: bool = True,
) -> list[dt.datetime]:
	"""
	Generate a list of dates with a frequency of 1 year. If `reset_times==True`, the month, day
	and time are ignored (only using the year values). This is useful if you want to include the
	first and last year even if they are not complete.

	Args:
		start_time: Start date for range
		end_time: End date for range
		reset_time: Whether to ignore the month, day and time values of the dates

	Returns:
		List of dates

	Example 1: Generate a range of dates with yearly frequency, ignoring months, days and time

		>>> start_time = dt.datetime(1994, 11, 5, 7, 23)
		>>> end_time = dt.datetime(1997, 2, 7, 0)
		>>> yearly_date_range(start_time, end_time)  # doctest: +NORMALIZE_WHITESPACE
		[datetime.datetime(1994, 1, 1, 0, 0),
		datetime.datetime(1995, 1, 1, 0, 0),
		datetime.datetime(1996, 1, 1, 0, 0),
		datetime.datetime(1997, 1, 1, 0, 0)]

	Example 2: Generate a range of dates with yearly frequency, including months, days and time

		>>> start_time = dt.datetime(1994, 11, 5, 7, 23)
		>>> end_time = dt.datetime(1997, 2, 7, 0)
		>>> yearly_date_range(start_time, end_time, reset_time=False)  # doctest: +NORMALIZE_WHITESPACE
		[datetime.datetime(1994, 11, 5, 7, 23),
		datetime.datetime(1995, 11, 5, 7, 23),
		datetime.datetime(1996, 11, 5, 7, 23)]
	"""
	if reset_time:
		# Only keep year values
		start_time = dt.datetime(start_time.year, 1, 1)
		end_time = dt.datetime(end_time.year, 1, 1)
	return list(rrule.rrule(freq=rrule.YEARLY, dtstart=start_time, until=end_time))


def select_time_range(data, start_time=None, end_time=None):
	"""
	Select a time range in data for pd.DataFrame, pd.Series, xr.DataArray, xr.Dataset
	"""
	if isinstance(data, xr.DataArray | xr.Dataset):
		return data.sel(time=slice(start_time, end_time))
	elif isinstance(data, pd.DataFrame):
		return data.loc[pd.IndexSlice[start_time:end_time, :], :]
	elif isinstance(data, pd.Series):
		return data.loc[pd.IndexSlice[start_time:end_time, :]]
