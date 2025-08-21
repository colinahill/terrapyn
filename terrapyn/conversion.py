import numpy as np

import terrapyn as tp


def celsius_to_kelvin(celsius):
	"""
	Convert temperature in degrees Celsius to degrees Kelvin.

	Args:
		celsius: Degrees Celsius

	Returns:
		Degrees Kelvin
	"""
	return celsius + 273.15


def kelvin_to_celsius(kelvin):
	"""
	Convert temperature in degrees Kelvin to degrees Celsius.

	Args:
		kelvin: Degrees Kelvin

	Returns:
		Degrees Celsius
	"""
	return kelvin - 273.15


def degrees_to_radians(degrees):
	"""
	Convert angular degrees to radians

	Args:
		degrees: Value in degrees to be converted.

	Returns:
		Value in radians
	"""
	return degrees * (np.pi / 180.0)


def radians_to_degrees(radians):
	"""
	Convert radians to angular degrees

	Args:
		radians: Value in radians to be converted.

	Returns:
		Value in angular degrees
	"""
	return radians * (180.0 / np.pi)


def kilometers_per_hour_to_meters_per_second(speed):
	"""
	Convert km/h to m/s.

	Args:
		speed: Speed in km/h

	Returns:
		Speed in m/s
	"""
	return speed / 3.6


def wind_speed_2m(ws, z):
	"""
	Convert wind speed measured at different heights above the soil surface to wind speed at 2 m above the surface,
	assuming a short grass surface. Based on FAO equation 47 in Allen et al (1998).

	Args:
		ws: Measured wind speed [m s-1]
		z: Height of wind measurement above ground surface [m]

	Returns:
		Wind speed at 2 m above the surface [m s-1]
	"""
	if ws is None:
		raise ValueError("`ws` must be provided")
	if z is None:
		raise ValueError("`z` must be provided")

	tp.validation.check_positive(ws, string="wind speed")
	tp.validation.check_positive(z, string="height of wind above surface")
	return (ws * 4.87) / np.log((67.8 * z) - 5.42)


def saturation_vapour_pressure_from_temperature(t, version=1):
	"""
	Estimate saturation vapour pressure (**es**) from air temperature.

	`version=1` based on Alduchov & Eskridge 1996: Improved Magnus' form approximation of
	saturation vapor pressure. J. Appl. Meteor., 35, 601–609.

	`version=2` is based on equations 11 and 12 in Allen et al (1998).

	Args:
		t: Temperature [deg C]
		version: Version of approximation to use.

	Returns:
		Saturation vapour pressure [kPa]
	"""
	if t is None:
		raise ValueError("`t` must be provided")

	if version == 1:
		return 0.61094 * np.exp((17.625 * t) / (t + 243.04))
	elif version == 2:
		return 0.6108 * np.exp((17.27 * t) / (t + 237.3))
	else:
		raise ValueError("`version` must be 1 or 2")


def delta_saturation_vapour_pressure(t):
	"""
	Estimate the slope of the saturation vapour pressure curve at a given temperature.
	Based on equation 13 in Allen et al (1998). If using in the Penman-Monteith *t* should be the
	mean air temperature.

	Args:
		t: Air temperature [deg C]. Use mean air temperature for use in Penman-Monteith.

	Returns:
		Saturation vapour pressure [kPa degC-1]
	"""
	if t is None:
		raise ValueError("`t` must be provided")

	numerator = 4098 * saturation_vapour_pressure_from_temperature(t)
	demoninator = np.power((t + 237.3), 2)
	return numerator / demoninator


def relative_humidity_from_actual_vapour_pressure_and_saturated_vapour_pressure(avp, svp):
	"""
	Calculate relative humidity from the ratio of actual vapour pressure to saturation vapour pressure
	at the same temperature. See Allen et al (1998), page 67 for details.

	Args:
		avp: Actual vapour pressure [kPa]
		svp: Saturated vapour pressure [kPa]

	Returns:
		Relative humidity [%]
	"""
	return 100.0 * avp / svp


def relative_humidity_from_dew_point_and_temperature(tdew, t):
	"""
	Calculate relative humidity from the dew point temperature and temperature.
	Converts the temperatures to vapour pressures using `saturation_vapour_pressure_from_temperature`.

	Args:
		tdew: Dew point temperature [deg C]
		t: Air temperature [deg C]

	Returns:
		Relative humidity [%]
	"""
	avp = saturation_vapour_pressure_from_temperature(tdew)
	svp = saturation_vapour_pressure_from_temperature(t)
	return relative_humidity_from_actual_vapour_pressure_and_saturated_vapour_pressure(avp, svp)


def psychrometric_constant_of_psychrometer(psychrometer=None, atmos_pres=None):
	"""
	Calculate the psychrometric constant for different types of psychrometer at a given atmospheric pressure.
	Based on FAO equation 16 in Allen et al (1998).

	Args:
		psychrometer: Integer between 1 and 3 which denotes type of psychrometer:

			1. ventilated (Asmann or aspirated type) psychrometer with an air movement of approximately 5 m/s
			2. natural ventilated psychrometer with an air movement of approximately 1 m/s
			3. non ventilated psychrometer installed indoors

		atmos_pres: Atmospheric pressure [kPa]. Can be estimated using ``atmospheric_pressure()``.

	Returns:
		Psychrometric constant [kPa degC-1].
	"""
	# Select coefficient based on type of ventilation of the wet bulb
	if psychrometer == 1:
		psy_coeff = 0.000662
	elif psychrometer == 2:
		psy_coeff = 0.000800
	elif psychrometer == 3:
		psy_coeff = 0.001200
	else:
		raise ValueError(f"psychrometer should be in range 1 to 3: {psychrometer}")

	if atmos_pres is None:
		raise ValueError("`atmos_pres` must be provided")

	return psy_coeff * atmos_pres


def psychrometric_constant(atmos_pres=None, altitude=None):
	"""
	Calculate the psychrometric constant. This method assumes that the air is saturated with water vapour at the
	minimum daily temperature. This assumption may not hold in arid areas.
	Based on equation 8, page 95 in Allen et al (1998).

	Args:
		atmos_pres: Atmospheric pressure [kPa]. Not required if `altitude` is provided.
		altitude: Elevation/altitude above sea level [m]. Not required if `atmos_pres` is provided.

	Returns:
		Psychrometric constant [kPa degC-1].
	"""
	if atmos_pres is None:
		if altitude is None:
			raise ValueError("`atmos_pres` or `altitude` must be provided")
		else:
			atmos_pres = atmospheric_pressure(altitude)
	tp.validation.check_positive(atmos_pres, "atmospheric pressure")
	return 0.000665 * atmos_pres


def atmospheric_pressure(altitude):
	"""
	Estimate atmospheric pressure from altitude. Calculated using a simplification of the ideal gas law,
	assuming 20 degrees Celsius for a standard atmosphere. Based on equation 7, page 62 in Allen et al (1998).

	Args:
		altitude: Elevation/altitude above sea level [m]

	Returns:
		atmospheric pressure [kPa]
	"""
	if altitude is None:
		raise ValueError("`altitude` must be provided")
	tmp = (293.0 - (0.0065 * altitude)) / 293.0
	return np.power(tmp, 5.26) * 101.3


def mean_saturation_vapour_pressure(tmin, tmax):
	"""
	Estimate mean saturation vapour pressure, *es* [kPa] from minimum and maximum temperature.
	Based on equations 11 and 12 in Allen et al (1998). Mean saturation vapour pressure is
	calculated as the mean of the saturation vapour pressure at tmax (maximum temperature)
	and tmin (minimum temperature).

	Args:
		tmin: Minimum temperature [deg C]
		tmax: Maximum temperature [deg C]

	Returns:
		Mean saturation vapour pressure (*es*) [kPa]
	"""
	if tmax is None:
		raise ValueError("`tmax` must be provided")
	if tmin is None:
		raise ValueError("`tmin` must be provided")

	return (saturation_vapour_pressure_from_temperature(tmin) + saturation_vapour_pressure_from_temperature(tmax)) / 2.0


def actual_vapour_pressure(
	tdew=None,
	twet=None,
	tdry=None,
	atmos_pres=None,
	psychrometer=None,
	tmin=None,
	tmax=None,
	rh_min=None,
	rh_max=None,
	rh_mean=None,
):
	"""
	Estimate actual vapour pressure [kPa] (*ea*) from multiple methods, in order of preference, based on the
	provided input parameters:

	1 - Based on `tdew` - If dewpoint temperature data are available.
	2 - Based on `twet`, `tdry`, `atmos_pres` - If dry and wet bulb temperatures are available from a psychrometer.
	3 - Based on `tmin`, `tmax`, `rh_min`, `rh_max` - If  minimum and maximum relative humidity data available.
	4 - Based on `tmin`, `rh_max` - If measurement errors of relative humidity are large then use only
	maximum relative humidity	.
	5 - Based on `tmin`, `tmax`, `rh_mean` - If minimum and maximum relative humidity are not available but mean
	relative humidity is available (less reliable than options 3 or 4).
	6 - Based on `tmin` - If no data for the above. This function is less reliable in arid areas where it is
	recommended that 2 degrees Celsius is subtracted from the minimum temperature before it is passed to the
	function (following advice given in Annex 6 of Allen et al (1998).

	Args:
		tdew: Dewpoint temperature [deg C].
		twet: Wet bulb temperature [deg C].
		tdry: Dry bulb temperature [deg C].
		atmos_pres: Atmospheric pressure [kPa]
		psychrometer: Integer between 1 and 3 which denotes type of psychrometer -
		see `psychrometric_constant_of_psychrometer()`.
		tmin: Daily minimum temperature [deg C]
		tmax: Daily maximum temperature [deg C].
		rh_min: Minimum relative humidity [%]
		rh_max: Maximum relative humidity [%]
		rh_mean: Mean relative humidity [%] (mean of RH min and RH max).

	Returns:
		Actual vapour pressure [kPa]
	"""
	if tdew is not None:
		return actual_vapour_pressure_from_tdew(tdew)
	elif all(i is not None for i in [twet, tdry, atmos_pres]):
		return actual_vapour_pressure_from_twet_tdry(twet, tdry, atmos_pres=atmos_pres, psychrometer=psychrometer)
	elif all(i is not None for i in [tmin, tmax, rh_min, rh_max]):
		return actual_vapour_pressure_from_rhmin_rhmax(rh_min=rh_min, rh_max=rh_max, tmin=tmin, tmax=tmax)
	elif all(i is not None for i in [tmin, rh_max]):
		return actual_vapour_pressure_from_rhmax(rh_max=rh_max, tmin=tmin)
	elif all(i is not None for i in [tmin, tmax, rh_mean]):
		return actual_vapour_pressure_from_rhmean(rh_mean=rh_mean, tmin=tmin, tmax=tmax)
	elif tmin is not None:
		return actual_vapour_pressure_from_tmin(tmin)
	else:
		raise ValueError("at least `tmin` must be provided")


def actual_vapour_pressure_from_tmin(tmin):
	"""
	Estimate actual vapour pressure (*ea*) from minimum temperature. This method is to be used where humidity data
	are lacking or are of questionable quality. The method assumes that the dewpoint temperature is approximately
	equal to the minimum temperature (*tmin*), i.e. the air is saturated with water vapour at *tmin*.
	**Note**: This assumption may not hold in arid/semi-arid areas.
	In these areas it may be better to subtract 2 deg C from the minimum temperature (see Annex 6 in FAO paper).
	Based on equation 48 in Allen et al (1998).

	Args:
		tmin: Daily minimum temperature [deg C]

	Returns:
		Actual vapour pressure [kPa]
	"""
	if tmin is None:
		raise ValueError("`tmin` must be provided")

	return saturation_vapour_pressure_from_temperature(tmin)


def actual_vapour_pressure_from_rhmin_rhmax(
	svp_tmin=None, svp_tmax=None, rh_min=None, rh_max=None, tmin=None, tmax=None
):
	"""
	Estimate actual vapour pressure (*ea*) from saturation vapour pressure and relative humidity.
	For periods of a week, ten days or a month, `rh_max` and `rh_min` are obtained by dividing
	the sum of the daily values by the number of days in that period.
	Based on FAO equation 17 in Allen et al (1998).

	Args:
		svp_tmin: Saturation vapour pressure at daily minimum temperature [kPa]. Not required if `tmin` is provided.
		svp_tmax: Saturation vapour pressure at daily maximum temperature [kPa]. Not required if `tmax` is provided.
		rh_min: Minimum relative humidity [%]
		rh_max: Maximum relative humidity [%]
		tmin: Daily minimum temperature [deg C]. Required if `svp_tmin` is not provided.
		tmax: Daily maximum temperature [deg C]. Required if `svp_tmax` is not provided.

	Returns:
		Actual vapour pressure [kPa]
	"""
	# Use `svp` if provided, otherwise use `tmin` and `tmax` to calculate `svp`
	if svp_tmin is None:
		if tmin is None:
			raise ValueError("`svp_tmin` or `tmin` must be provided")
		else:
			svp_tmin = saturation_vapour_pressure_from_temperature(tmin)
	if svp_tmax is None:
		if tmax is None:
			raise ValueError("`svp_tmax` or `tmax` must be provided")
		else:
			svp_tmax = saturation_vapour_pressure_from_temperature(tmax)
	tmp1 = svp_tmin * rh_max / 100.0
	tmp2 = svp_tmax * rh_min / 100.0
	return (tmp1 + tmp2) / 2.0


def actual_vapour_pressure_from_rhmax(svp_tmin=None, rh_max=None, tmin=None):
	"""
	Estimate actual vapour pressure (*ea*) from saturation vapour pressure at daily minimum temperature
	and maximum relative humidity. Based on FAO equation 18 in Allen et al (1998).

	Args:
		svp_tmin: Saturation vapour pressure at daily minimum temperature [kPa]. Not required if `tmin` is provided.
		rh_max: Maximum relative humidity [%]
		tmin: Daily minimum temperature [deg C]. Required if `svp_tmin` is not provided.

	Returns:
		Actual vapour pressure [kPa]
	"""
	if svp_tmin is None:
		if tmin is None:
			raise ValueError("`svp_tmin` or `tmin` must be provided")
		else:
			svp_tmin = saturation_vapour_pressure_from_temperature(tmin)

	return svp_tmin * rh_max / 100.0


def actual_vapour_pressure_from_rhmean(svp_tmin=None, svp_tmax=None, rh_mean=None, tmin=None, tmax=None):
	"""
	Estimate actual vapour pressure (*ea*) from saturation vapour pressure at daily minimum and maximum
	temperature, and mean relative humidity. Based on FAO equation 19 in Allen et al (1998).

	Args:
		svp_tmin: Saturation vapour pressure at daily minimum temperature [kPa]. Not required if `tmin` is provided.
		svp_tmax: Saturation vapour pressure at daily maximum temperature [kPa]. Not required if `tmax` is provided.
		rh_mean: Mean relative humidity [%] (average of RH min and RH max).
		tmin: Daily minimum temperature [deg C]. Required if `svp_tmin` is not provided.
		tmax: Daily maximum temperature [deg C]. Required if `svp_tmax` is not provided.

	Returns:
		Actual vapour pressure [kPa]
	"""
	# Use `svp` if provided, otherwise use `tmin` and `tmax` to calculate `svp`
	if svp_tmin is None:
		if tmin is None:
			raise ValueError("`svp_tmin` or `tmin` must be provided")
		else:
			svp_tmin = saturation_vapour_pressure_from_temperature(tmin)
	if svp_tmax is None:
		if tmax is None:
			raise ValueError("`svp_tmax` or `tmax` must be provided")
		else:
			svp_tmax = saturation_vapour_pressure_from_temperature(tmax)
	return (rh_mean / 100.0) * ((svp_tmax + svp_tmin) / 2.0)


def actual_vapour_pressure_from_tdew(tdew):
	"""
	Estimate actual vapour pressure (*ea*) from dewpoint temperature.
	Based on equation 14 in Allen et al (1998). As the dewpoint temperature is the temperature to which air
	needs to be cooled to make it saturated, the actual vapour pressure is the saturation vapour pressure at
	the dewpoint temperature. This method is preferable to calculating vapour pressure from minimum temperature.

	Args:
		tdew: Dewpoint temperature [deg C]

	Returns:
		Actual vapour pressure [kPa]
	"""
	if tdew is None:
		raise ValueError("`tdew` must be provided")

	return saturation_vapour_pressure_from_temperature(tdew)


def actual_vapour_pressure_from_twet_tdry(
	twet, tdry, svp_twet=None, psy_const=None, psychrometer=None, atmos_pres=None
):
	"""
	Estimate actual vapour pressure (*ea*) from wet and dry bulb temperature.
	Based on equation 15 in Allen et al (1998). As the dewpoint temperature is the temperature to which air needs
	to be cooled to make it saturated, the actual vapour pressure is the saturation vapour pressure at the dewpoint
	temperature. This method is preferable to calculating vapour pressure from minimum temperature.
	Values for the psychrometric constant of the psychrometer ``psy_const`` can be calculated using
	``psyc_const_of_psychrometer()``.

	Args:
		twet: Wet bulb temperature [deg C]
		tdry: Dry bulb temperature [deg C]
		svp_twet: Saturated vapour pressure at the wet bulb temperature [kPa]. Not required if `twet` is provided.
		psy_const: Psychrometric constant of the pyschrometer [kPa deg C-1].  Not required if both `atmos_pres` and
		`psychrometer` or just `atmos_pres` are provided.
		psychrometer: Integer between 1 and 3 which denotes type of psychrometer
		- see `psychrometric_constant_of_psychrometer()`
		atmos_pres: Atmospheric pressure [kPa]

	Returns:
		Actual vapour pressure [kPa]
	"""
	if twet is None:
		raise ValueError("`twet` must be provided")
	if tdry is None:
		raise ValueError("`tdry` must be provided")

	if svp_twet is None:
		svp_twet = saturation_vapour_pressure_from_temperature(twet)

	if psy_const is None:
		if atmos_pres is None:
			raise ValueError("`atmos_pres` must be provided")
		if psychrometer is None:
			psy_const = psychrometric_constant(atmos_pres)
		else:
			psy_const = psychrometric_constant_of_psychrometer(psychrometer, atmos_pres)
	return svp_twet - psy_const * (tdry - twet)


def energy_to_evaporation(energy):
	"""
	Convert energy (e.g. radiation energy) in MJ m-2 day-1 to the equivalent evaporation, assuming
	a grass reference crop. Energy is converted to equivalent evaporation using a conversion factor
	equal to the inverse of the latent heat of vapourisation (1 / lambda = 0.408).
	Based on FAO equation 20 in Allen et al (1998).

	Args:
		energy: Energy e.g. radiation or heat flux [MJ m-2 day-1].

	Returns:
		Equivalent evaporation [mm day-1].
	"""
	if energy is None:
		raise ValueError("`energy` must be provided")

	return 0.408 * energy


def monthly_soil_heat_flux(t_month_prev, t_month, next_month=False):
	"""
	Estimate monthly soil heat flux (Gmonth) from the mean air temperature of the previous and current or
	next month, assuming a grass crop.
	When `next_month==True`, `t_month` is for the next month, based on equation 43 in Allen et al (1998).
	WHen `next_month==False` `t_month` is for the current month, based on equation 44 in Allen et al (1998).
	The resulting heat flux can be converted to equivalent evaporation [mm day-1] using ``energy2evap()``.

	Args:
		t_month_prev: Mean air temperature of the previous month [deg Celsius]
		t_month: Mean air temperature of the current/next month [deg Celsius]
		next_month: If `True` then `t_month` is assumed to be the mean temperature of the next month,
		otherwise `t_month` is assumed to be the mean for the current month.

	Returns:
		Monthly soil heat flux (Gmonth) [MJ m-2 day-1]
	"""
	factor = 0.07 if next_month else 0.14
	return factor * (t_month - t_month_prev)


def vectors_to_scalar(u, v):
	"""Compute the scalar (length) from u and v vector components"""
	return np.sqrt(u * u + v * v)


def normalize_angle(angle_rad):
	"""
	Return the input angle values in the range [0, 2 * pi].

	Args:
		angle_rad: Input angle in radians.

	Returns:
		Angle values in the range [0, 2 * pi]
	"""
	return angle_rad % (2 * np.pi)


def vectors_to_angle(u, v):
	"""Return the angle between two vectors in radians"""
	return np.arctan2(v, u)


def angle_to_bearing(angle, convention: str = "from", unit: str = "deg"):
	"""
	Convert angle in radians to compass bearing convention, where the direction can be 'from'
	(the meteorological convention) or 'to' (the oceanographic convention).
	If angle is zero, bearing is set to 0 by convention.

	Args:
		angle: Angle values in radians to be converted from trigonometric to compass bearing convention.
		convention: Convention to return direction where 'from' (default) returns the direction the wind is coming
		from (meteorological convention), and 'to' returns the direction the wind is going towards
		(oceanographic convention).
		unit: Unit of angle, either 'rad' or 'deg' (default).

	Returns:
		Angle values in the interval [0, 360) degrees or [0, 2 * pi) radians.
	"""
	if unit not in ("deg", "rad"):
		raise ValueError("Unit must be 'deg' or 'rad'.")

	if convention == "from":
		compass = -np.pi / 2 - angle
	elif convention == "to":
		compass = np.pi / 2 - angle
	else:
		raise ValueError("Convention must be 'from' or 'to'.")

	compass = normalize_angle(compass)

	if unit == "deg":
		return radians_to_degrees(compass)
	return compass


def wind_speed(u, v):
	"""
	Compute the wind speed (scalar) from U and V vector components.

	Args:
		u: U component of the wind based on the [ECMWF](https://apps.ecmwf.int/codes/grib/param-db/?id=131)
			definition.
		v: V component of wind based on the [ECMWF](https://apps.ecmwf.int/codes/grib/param-db?id=132)
			definition.
	Returns:
		Wind speed in the same unit as u and v.
	"""
	return vectors_to_scalar(u, v)


def wind_direction(u, v, convention: str = "from", unit: str = "deg"):
	"""
	Compute the wind direction from U and V components. If U=V=0, wind direction is set to 0 by convention.

	Args:
		u: U component of the wind based on the [ECMWF](https://apps.ecmwf.int/codes/grib/param-db/?id=131)
			definition.
		v: V component of wind based on the [ECMWF](https://apps.ecmwf.int/codes/grib/param-db?id=132)
			definition.
		convention: Convention to return direction where 'from' returns the direction the wind is coming
		from (meteorological convention), and 'to' returns the direction the wind is going towards
		(oceanographic convention).
		unit: Unit of wind direction, 'rad' or 'deg'.
	Returns:
		The direction of the wind in the interval [0, 360) degrees or [0, 2*pi) radians.
	"""
	zero_wind_mask = (np.abs(u) == 0) & (np.abs(v) == 0)
	angle = vectors_to_angle(u, v)
	bearing = angle_to_bearing(angle, convention=convention, unit=unit)
	bearing = np.where(zero_wind_mask, 0, bearing)
	return bearing


def cartesian_to_polar(x, y, compass: bool = False) -> tuple:
	"""
	Transform cartesian coordinates into polar coordinates (modulus and angle). Polar coordinates
	are given in the range [0, 2 * pi] using the trigonometric or bearing convention depending on
	**compass**.

	Args:
		x: Input values for x cartesian component.
		y: Input values for y cartesian component.
		compass: If True, angle values are given in bearing convention. If False (default), angle
		values are given in trigonometric convention.

	Returns:
		Tuple of modulus and angle values of the vector field given by **x** and **y**.
	"""
	modulus = np.sqrt(np.square(x) + np.square(y))

	angle = vectors_to_angle(y, x)

	if compass:
		return modulus, angle_to_bearing(angle)
	else:
		return modulus, normalize_angle(angle)
