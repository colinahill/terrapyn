import numpy as np

# Latitude
_MIN_LATITUDE_RADIANS = -1.5707963267948966
_MAX_LATITUDE_RADIANS = 1.5707963267948966

# Solar declination
_MIN_SOLAR_DECLINATION_RADIANS = -0.41015237421866746
_MAX_SOLAR_DECLINATION_RADIANS = 0.41015237421866746

# Sunset hour angle
_MIN_SUNSET_HOUR_ANGLE_RADIANS = 0.0
_MAX_SUNSET_HOUR_ANGLE_RADIANS = np.pi


def check_day_hours(hours):
	"""
	Check that `hours` is in the range 0 to 24.
	"""
	if hours is None:
		raise ValueError("`hours` must be provided")
	elif np.any(hours < 0) or np.any(hours > 24):
		raise ValueError("`hours` should be in range 0-24")


def check_day_of_year(day_of_year):
	"""
	Check day of the year is valid, between 1-366.
	"""
	if day_of_year is None:
		raise ValueError("`day_of_year` must be provided")
	elif np.any(day_of_year < 1) or np.any(day_of_year > 366):
		raise ValueError("Day of the year must be in range 1-366")


def check_latitude_radians(latitude):
	"""
	Check latitude is within range -pi/2 to +pi/2 radians (-90 to +90 degrees)
	"""
	if latitude is None:
		raise ValueError("`latitude` must be provided")
	elif np.any(latitude < _MIN_LATITUDE_RADIANS) or np.any(latitude > _MAX_LATITUDE_RADIANS):
		raise ValueError(f"latitude outside valid range {_MIN_LATITUDE_RADIANS} to {_MAX_LATITUDE_RADIANS}")


def check_solar_declination_radians(solar_declination):
	"""
	Solar declination can vary between -23.5 and +23.5 degrees (in radians).

	See http://mypages.iit.edu/~maslanka/SolarGeo.pdf
	"""
	if solar_declination is None:
		raise ValueError("`solar_declination` must be provided")
	elif np.any(solar_declination < _MIN_SOLAR_DECLINATION_RADIANS) or np.any(
		solar_declination > _MAX_SOLAR_DECLINATION_RADIANS
	):
		raise ValueError(
			f"solar declination outside valid range {_MIN_SOLAR_DECLINATION_RADIANS} to "
			f"{_MAX_SOLAR_DECLINATION_RADIANS}"
		)


def check_sunset_hour_angle_radians(sunset_hour_angle):
	"""
	Sunset hour angle has the range 0 to pi radians (0 to 180 degrees).

	See http://mypages.iit.edu/~maslanka/SolarGeo.pdf
	"""
	if sunset_hour_angle is None:
		raise ValueError("`sunset_hour_angle` must be provided")
	elif np.any(sunset_hour_angle < _MIN_SUNSET_HOUR_ANGLE_RADIANS) or np.any(
		sunset_hour_angle > _MAX_SUNSET_HOUR_ANGLE_RADIANS
	):
		raise ValueError(
			f"`sunset_hour_angle` outside valid range {_MIN_SUNSET_HOUR_ANGLE_RADIANS} to "
			f"{_MAX_SUNSET_HOUR_ANGLE_RADIANS}"
		)


def check_positive(values, string=None):
	"""
	Check values are positive.

	Args:
		values: Array-like values to check.
		string: Optional string to pass to the raised value error, in the case of non-positive values.

	Returns:
		`True` if `values` is contains only positive numbers, otherwise `False`.
	"""
	if values is None:
		raise ValueError("`values` must be provided")
	elif np.any(np.isnan(values)):
		raise ValueError("`values` contains `NaN`")
	elif np.any(values < 0.0):
		if string is not None:
			raise ValueError(f"{string} has some non-positive values")
		else:
			raise ValueError("Non-positive values")
