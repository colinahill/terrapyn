import numpy as np

from terrapyn import conversion
from terrapyn.validation import check_day_hours as _check_day_hours
from terrapyn.validation import check_day_of_year as _check_day_of_year
from terrapyn.validation import check_latitude_radians as _check_latitude_radians
from terrapyn.validation import check_solar_declination_radians as _check_solar_declination_radians
from terrapyn.validation import check_sunset_hour_angle_radians as _check_sunset_hour_angle_radians

# Solar constant
SOLAR_CONSTANT = 117.57312  # [ MJ m-2 day-1] = 1360.8 W/m2

# Stefan Boltzmann constant
STEFAN_BOLTZMANN_CONSTANT = 0.000000004899203  # [MJ K-4 m-2 day-1]


def _solar_parameters(latitude=None, day_of_year=None, sunshine_hours=None, altitude=None, coastal=False):
	"""
	Calculate and return solar declination, sunset hour angle, inverse earth-sun distance,
	and extraterrestrial radiation (helper function, as these parameters are commonly used together).

	Args:
		latitude: Latitude [radians]
		day_of_year: Day of year, between 1 and 366.
		sunshine_hours: Total sunshine duration for that latitude, for that day [hours].
		altitude: Altitude/elevation of point above sea level [m]
		coastal: ``True`` if site is a coastal location, situated on or adjacent to coast of a large land mass
		and where air masses are influenced by a nearby water body, ``False`` if interior location where land
		mass dominates and air masses are not strongly influenced by a large water body.

	"""
	if latitude is None:
		raise ValueError("'latitude' is not given")
	if day_of_year is None:
		raise ValueError("'day_of_year' is not given")

	sol_dec = solar_declination(day_of_year)
	sha = sunset_hour_angle(latitude, sol_dec)
	ird = inverse_relative_distance_earth_sun(day_of_year)
	et_rad = extraterrestrial_radiation(latitude, sol_dec, sha, ird)
	return (
		sol_dec,
		sha,
		ird,
		et_rad,
	)


def clear_sky_radiation(altitude=None, et_rad=None):
	"""
	Estimate clear sky radiation from altitude and extraterrestrial radiation.
	Based on equation 37 in Allen et al (1998) which is recommended when calibrated Angstrom values
	are not available.

	Args:
		altitude: Elevation above sea level [m]
		et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using ``extraterrestrial_radiation()``.

	Returns:
		Clear sky radiation [MJ m-2 day-1]
	"""
	if altitude is None:
		raise ValueError("`altitude` must be provided")
	if et_rad is None:
		raise ValueError("`et_rad` must be provided")
	return (0.00002 * altitude + 0.75) * et_rad


def total_incoming_shortwave_radiation_from_sunshine_hours(dl_hours=None, sunshine_hours=None, et_rad=None):
	"""
	Calculate incoming shortwave radiation `Rs` (radiation hitting a horizontal plane
	after scattering by the atmosphere) from relative sunshine duration. If measured radiation data
	are not available, this method is preferable to calculating solar radiation from temperature.
	If a monthly mean is required, then divide the monthly number of sunshine hours by number of
	days in the month and ensure that `et_rad` and `dl_hours` was calculated using the day of the
	year that corresponds to the middle of the month.
	Based on equations 34 and 35 in Allen et al (1998).

	Args:
		dl_hours: Number of daylight hours [hours]. Can be calculated using `daylight_hours()`.
		sunshine_hours: Sunshine duration [hours]. If not available, use
		`total_incoming_shortwave_radiation_from_temperature()` instead of this function.
		et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using `extraterrestrial_radiation()`.

	Returns:
		Incoming solar (or shortwave) radiation [MJ m-2 day-1]
	"""
	for variable, name in zip(
		[dl_hours, sunshine_hours, et_rad], ["dl_hours", "sunshine_hours", "et_rad"], strict=False
	):
		if variable is None:
			raise ValueError(f"`{name}` must be provided")

	_check_day_hours(sunshine_hours)
	_check_day_hours(dl_hours)

	# 0.5 and 0.25 are default values of regression constants (Angstrom values)
	# recommended by FAO when calibrated values are unavailable.
	return (0.5 * sunshine_hours / dl_hours + 0.25) * et_rad


def total_incoming_shortwave_radiation_from_temperature(
	et_rad=None, cs_rad=None, altitude=None, tmin=None, tmax=None, coastal=False
):
	"""
	Estimate incoming shortwave radiation `Rs` (radiation hitting a horizontal plane after scattering by the
	atmosphere) from min and max temperature together with an empirical adjustment coefficient for 'interior'
	and 'coastal' regions. The formula is based on equation 50 in Allen et al (1998) which is the
	Hargreaves radiation formula (Hargreaves and Samani, 1982, 1985). This method should be used
	only when solar radiation or sunshine hours data are not available. It is only recommended for locations where
	it is not possible to use radiation data from a regional station (either because climate conditions are
	heterogeneous or data are lacking).
	**NOTE**: this method is not suitable for island locations due to the moderating effects of the surrounding water.

	Args:
		et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using `extraterrestrial_radiation()`.
		cs_rad: Clear sky radiation [MJ m-2 day-1]. If not available, can provide `altitude` instead.
		altitude: Altitude/elevation of point above sea level [m]. Only required if `cs_rad` is not given.
		tmin: Daily minimum temperature [deg C].
		tmax: Daily maximum temperature [deg C].
		coastal: ``True`` if site is a coastal location, situated on or adjacent to coast of a large land mass
		and where air masses are influenced by a nearby water body, ``False`` if interior location where land
		mass dominates and air masses are not strongly influenced by a large water body.

	Returns:
		Incoming solar (or shortwave) radiation (Rs) [MJ m-2 day-1].
	"""
	for variable, name in zip([tmax, tmin, et_rad], ["tmax", "tmin", "et_rad"], strict=False):
		if variable is None:
			raise ValueError(f"`{name}` must be provided")

	if cs_rad is None:
		cs_rad = clear_sky_radiation(altitude, et_rad)

	# Determine value of adjustment coefficient [deg C-0.5] for coastal/interior locations
	adj = 0.19 if coastal else 0.16

	total_incoming_shortwave_rad = adj * np.sqrt(tmax - tmin) * et_rad

	# The solar radiation value is constrained by the clear sky radiation
	return np.minimum(total_incoming_shortwave_rad, cs_rad)


def total_incoming_shortwave_radiation(
	dl_hours=None,
	sunshine_hours=None,
	et_rad=None,
	cs_rad=None,
	altitude=None,
	tmin=None,
	tmax=None,
	coastal=False,
):
	"""
	Calculate the total incoming shortwave radiation per day `Rs` (radiation hitting a horizontal plane after
	scattering by the atmosphere) in units of [MJ m-2 day-1].

	This is a wrapper that automatically chooses between two other functions:
	`total_incoming_shortwave_radiation_from_sunshine_hours()` or
	`total_incoming_shortwave_radiation_from_temperature()`.

	The method of estimation is chosen automatically depending on which parameters are given, where
	the preferred option is to provide `dl_hours`, `sunshine_hours`, and `et_rad`. Otherwise, `et_rad`,
	`cs_rad`/`altitude`, `tmin`, `tmax`, and `coastal` should be provided.

	If a monthly mean is required, divide the monthly number of sunshine hours by number of days in the
	month and ensure that `et_rad` and `dl_hours` are calculated using the day of the year that
	corresponds to the middle of the month.

	Args:
		dl_hours: Number of daylight hours [hours]. Can be calculated using `daylight_hours()`.
		sunshine_hours: Sunshine duration [hours].
		altitude: Altitude/elevation of point above sea level [m]
		et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using `extraterrestrial_radiation()`.
		cs_rad: Clear sky radiation [MJ m-2 day-1]. Can be estimated using `clear_sky_radiation()`.
		tmin: Daily minimum temperature [deg C].
		tmax: Daily maximum temperature [deg C].
		coastal: ``True`` if site is a coastal location, situated on or adjacent to coast of a large land mass
		and where air masses are influenced by a nearby water body, ``False`` if interior location where land
		mass dominates and air masses are not strongly influenced by a large water body.

	Returns:
		Incoming solar (or shortwave) radiation (Rs) [MJ m-2 day-1].
	"""
	# If either of df_hours and sunshine_hours are not provided, estimate total incoming shortwave
	# radiation from temperature
	if None in [dl_hours, sunshine_hours]:
		total_incoming_shortwave_rad = total_incoming_shortwave_radiation_from_temperature(
			et_rad=et_rad, cs_rad=cs_rad, altitude=altitude, tmin=tmin, tmax=tmax, coastal=coastal
		)

	# Else estimate total incoming shortwave radiation from sunshine hours and daylight hours
	else:
		total_incoming_shortwave_rad = total_incoming_shortwave_radiation_from_sunshine_hours(
			dl_hours, sunshine_hours, et_rad
		)

	return total_incoming_shortwave_rad


def net_incoming_shortwave_radiation(
	total_incoming_shortwave_rad=None,
	albedo=0.23,
	sunshine_hours=None,
	dl_hours=None,
	et_rad=None,
	cs_rad=None,
	altitude=None,
	tmin=None,
	tmax=None,
	coastal=False,
):
	"""
	Calculate net incoming shortwave radiation from gross (total) incoming shortwave solar radiation,
	and the albedo (reflectance) from the surface. Net incoming shortwave radiation is equal to the incoming shortwave
	radiation minus the radiation lost from reflected solar radiation. The output can be converted to
	equivalent evaporation [mm day-1] using `energy_to_evaporation()`.

	The default value of `albedo=0.23` assumes a grass reference crop. Based on FAO equation 38 in Allen et al (1998).

	Args:
		total_incoming_shortwave_rad: Total incoming shortwave radiation [MJ m-2 day-1]. If given, no other arguments
		are required, apart from `albedo`. If not provided, this is estimated by passing all other arguments of
		this function (excluding `albedo`) to `total_incoming_shortwave_radiation()`.
		albedo: Albedo of the crop as the proportion of gross incoming solar radiation that is reflected by the
		surface. Default value is 0.23, which is the value used by the FAO for a short grass reference crop. Albedo
		can be as high as 0.95 for freshly fallen snow and as low as 0.05 for wet bare soil. A green vegetation over
		has an albedo of about 0.20-0.25 (Allen et al, 1998).
		latitude: Latitude [radians] (Not required if `total_incoming_shortwave_rad` is provided).
		day_of_year: Day of year, between 1 and 366. (Not required if `total_incoming_shortwave_rad` is provided).
		sunshine_hours: Total sunshine duration for that latitude, for that day [hours]. (Not required if
		`total_incoming_shortwave_rad` is provided).
		altitude: Altitude/elevation of point above sea level [m]. (Not required if `total_incoming_shortwave_rad`
		is provided).
		coastal: ``True`` if site is a coastal location, situated on or adjacent to coast of a large land mass
		and where air masses are influenced by a nearby water body, ``False`` if interior location where land
		mass dominates and air masses are not strongly influenced by a large water body. (Not required if
		`total_incoming_shortwave_rad` is provided).

	Returns:
		Net incoming shortwave radiation [MJ m-2 day-1] (net incoming minus reflected).
	"""
	if total_incoming_shortwave_rad is None:
		total_incoming_shortwave_rad = total_incoming_shortwave_radiation(
			dl_hours=dl_hours,
			sunshine_hours=sunshine_hours,
			et_rad=et_rad,
			cs_rad=cs_rad,
			altitude=altitude,
			tmin=tmin,
			tmax=tmax,
			coastal=coastal,
		)
	return (1 - albedo) * total_incoming_shortwave_rad


def net_outward_longwave_radiation(
	tmin=None,
	tmax=None,
	total_incoming_shortwave_rad=None,
	et_rad=None,
	cs_rad=None,
	avp=None,
	latitude=None,
	day_of_year=None,
	dl_hours=None,
	sunshine_hours=None,
	altitude=None,
	coastal=False,
	tdew=None,
	twet=None,
	tdry=None,
	rh_min=None,
	rh_max=None,
	atmos_pres=None,
	rh_mean=None,
):
	"""
	Estimate net outgoing longwave radiation. This is the net longwave energy (net energy flux) leaving the
	earth's surface. It is proportional to the absolute temperature of the surface raised to the fourth power
	according to the Stefan-Boltzmann law. However, water vapour, clouds, carbon dioxide and dust are absorbers
	and emitters of longwave radiation. This function corrects the Stefan-Boltzmann law for humidity
	(using actual vapor pressure) and cloudiness (using solar radiation and clear sky radiation). The concentrations
	of all other absorbers are assumed to be constant.
	The output can be converted to equivalent evaporation [mm day-1] using ``energy_to_evaporation()``.
	Based on FAO equation 39 in Allen et al (1998).

	Args:
		tmin: Daily minimum temperature [deg C]
		tmax: Daily maximum temperature [deg C]
		total_incoming_shortwave_rad: Total incoming shortwave radiation [MJ m-2 day-1]. Not required
		if `latitude`, `day_of_year`, `sunshine_hours`, `altitude`, `coastal` are provided.
		et_rad: Extraterrestrial radiation [MJ m-2 day-1]. Can be estimated using `extraterrestrial_radiation()`.
		cs_rad: Clear sky radiation [MJ m-2 day-1]. Optional, calculated automatically with other parameters.
		avp: Actual vapour pressure [kPa]. Can be estimated from multiple methods, in order of preference:

			1. If dewpoint temperature data are available, based on `tdew`.
			2. If dry and wet bulb temperatures are available from a psychrometer,
			based on `twet`, `tdry`, `atmos_pres`.
			3. If  minimum and maximum relative humidity data available, based on `tmin`,
			`tmax`, `rh_min`, `rh_max`.
			4. If measurement errors of relative humidity are large then use only maximum
			relative humidity, based on `tmin`, `rh_max`.
			5. If minimum and maximum relative humidity are not available but mean relative
			humidity is available, based on `tmin`, `tmax`, `rh_mean` (less reliable than options 3 or 4).
			6. If no data for the above, based on `tmin`. This function is less reliable in arid areas
			where it is recommended that 2 degrees Celsius is subtracted from the minimum temperature
			before it is passed to the function (following advice given in Annex 6 of Allen et al (1998).

		latitude: Latitude [radians] - should be negative if it in the southern hemisphere, positive if in
		the northern hemisphere.
		day_of_year: Day of year integer between 1 and 366.
		dl_hours: Number of daylight hours [hours]. Can be calculated using `daylight_hours()`.
		sunshine_hours: Sunshine duration [hours].
		altitude: Altitude/elevation of point above sea level [m]
		coastal: ``True`` if site is a coastal location, situated on or adjacent to coast of a large land mass
		and where air masses are influenced by a nearby water body, ``False`` if interior location where land
		mass dominates and air masses are not strongly influenced by a large water body.
		tdew: Dew point temperature [deg C].
		twet: Wet buld temperature [deg C].
		tdry: Dry air temperature [deg C].
		rh_min: Minimum relative humidity [%].
		rh_max: Maximum relative humidity [%].
		atmos_pres: Atmospheric pressure [kPa].
		rh_mean: Mean relative humidity [kPa].

	Return:
		Net outgoing longwave radiation [MJ m-2 day-1]
	"""
	if tmax is None:
		raise ValueError("'tmax' must be provided")
	if tmin is None:
		raise ValueError("'tmin' must be provided")

	if total_incoming_shortwave_rad is None:
		total_incoming_shortwave_rad = total_incoming_shortwave_radiation(
			dl_hours=dl_hours,
			sunshine_hours=sunshine_hours,
			et_rad=et_rad,
			cs_rad=cs_rad,
			altitude=altitude,
			tmin=tmin,
			tmax=tmax,
			coastal=coastal,
		)
	if cs_rad is None:
		sol_dec, sha, ird, et_rad = _solar_parameters(
			latitude=latitude,
			day_of_year=day_of_year,
			sunshine_hours=sunshine_hours,
			altitude=altitude,
			coastal=coastal,
		)
		cs_rad = clear_sky_radiation(altitude=altitude, et_rad=et_rad)

	if avp is None:
		avp = conversion.actual_vapour_pressure(
			tdew=tdew,
			twet=twet,
			tdry=tdry,
			atmos_pres=atmos_pres,
			tmin=tmin,
			tmax=tmax,
			rh_min=rh_min,
			rh_max=rh_max,
			rh_mean=rh_mean,
		)

	tmp1 = (np.power(conversion.celsius_to_kelvin(tmax), 4) + np.power(conversion.celsius_to_kelvin(tmin), 4)) / 2
	tmp2 = 0.34 - (0.14 * np.sqrt(avp))
	tmp3 = 1.35 * (total_incoming_shortwave_rad / cs_rad) - 0.35
	return STEFAN_BOLTZMANN_CONSTANT * tmp1 * tmp2 * tmp3


def daylight_hours(sha=None, latitude=None, sol_dec=None, day_of_year=None):
	"""
	Calculate daylight hours from sunset hour angle. Optionally, use latitude, and solar declination or day of year.
	Based on FAO equation 34 in Allen et al (1998).

	Args:
		sha: Sunset hour angle [rad]. Not required if `latitude` and `sol_dec`, or `day_of_year` are provided.
		latitude: Latitude [radians] -  should be negative if it in the southern hemisphere, positive if in
		the northern hemisphere. Not required if `sha` is provided.
		sol_dec: Solar declination [radians].  Not required if `day_of_year` or `sha` is provided.
		day_of_year: Day of year integer between 1 and 365 or 366. Not required if `sol_dec` or `sha` is provided.

	Returns:
		Daylight hours [h].
	"""
	if sha is None:
		if latitude is None:
			raise ValueError("Either `sha` or `latitude` must be provided")
		else:
			sha = sunset_hour_angle(latitude=latitude, sol_dec=sol_dec, day_of_year=day_of_year)
	else:
		_check_sunset_hour_angle_radians(sha)
	return (24.0 / np.pi) * sha


def sunset_hour_angle(latitude=None, sol_dec=None, day_of_year=None):
	"""
	Calculate sunset hour angle (*Ws*) from latitude and solar declination.
	Based on FAO equation 25 in Allen et al (1998).

	Args:
		latitude: Latitude [radians] -  should be negative if it in the southern hemisphere,
		positive if in the northern hemisphere.
		sol_dec: Solar declination [radians].  Not required if `day_of_year` is provided.

	Returns:
		Sunset hour angle [radians].
	"""
	_check_latitude_radians(latitude)

	if sol_dec is None:
		if day_of_year is None:
			raise ValueError("Either `sol_dec` or `day_of_year` must be provided")
		else:
			sol_dec = solar_declination(day_of_year)
	else:
		_check_solar_declination_radians(sol_dec)

	cos_sha = -np.tan(latitude) * np.tan(sol_dec)

	# Domain of arccos is -1 <= x <= 1 radians (this is not mentioned in FAO-56)
	# See https://itaca.stuffnting.com/the-sun-as-a-source-of-energy/part-3-calculating-solar-angles/
	# Ensure cos_sha values lie within the range -1 <= cos_sha <= 1

	# If cos_sha is >= 1 there is no sunset, i.e. 24 hours of daylight
	cos_sha_no_sunset = np.where(cos_sha > 1.0, 1, cos_sha)

	# If cos_sha is <= 1 there is no sunrise, i.e. 24 hours of darkness
	cos_sha_no_sunrise_sunset = np.where(cos_sha_no_sunset < -1.0, -1, cos_sha_no_sunset)
	return np.arccos(cos_sha_no_sunrise_sunset)


def solar_declination(day_of_year=None):
	"""
	Calculate solar declination from day of the year.
	Based on FAO equation 24 in Allen et al (1998).

	Args:
		day_of_year: Day of year integer between 1 and 365 or 366.

	Returns:
		solar declination [radians]
	"""
	_check_day_of_year(day_of_year)
	return 0.409 * np.sin((2.0 * np.pi / 365.0) * day_of_year - 1.39)


def extraterrestrial_radiation(latitude=None, sol_dec=None, sha=None, ird=None, day_of_year=None):
	# solar_time_start=None, solar_time_end=None
	"""
	Estimate daily extraterrestrial radiation (*Ra*, 'top of the atmosphere radiation').
	Based on equation 21 in Allen et al (1998). If monthly mean radiation is required make sure
	*sol_dec*, *sha* and *irl* have been calculated using the day of the year that corresponds to
	the middle of the month. **Note**: From Allen et al (1998): "For the winter months in latitudes
	greater than 55 degrees (N or S), the equations have limited validity. Reference should be made
	to the Smithsonian Tables to assess possible deviations."

	Args:
		latitude: Latitude [radians] - should be negative if in the southern hemisphere, positive
		if in the northern hemisphere.
		sol_dec: Solar declination [radians]. Not required if `day_of_year` is provided.
		sha: Sunset hour angle [radians]. Not required if `latitude` and `sol_dec` or `day_of_year` are provided.
		ird: Inverse relative distance earth-sun [dimensionless]. Not required if `day_of_year` is provided.
		day_of_year: Day of year (1-365/366). Not required if `ird` and `sol_dec` are provided.

	TODO:
		Add ability to calculate radiation for periods less than 1 day - see eqn 28 from Allen
		solar_time_start: Start solar time for periods less than 1 day.
		solar_time_end: End solar time for periods less than 1 day.

	Returns:
		Daily extraterrestrial radiation [MJ m-2 day-1]
	"""
	_check_latitude_radians(latitude)

	if ird is None:
		if day_of_year is None:
			raise ValueError("Either `ird` or `day_of_year` must be provided")
		else:
			ird = inverse_relative_distance_earth_sun(day_of_year)

	if sol_dec is None:
		if day_of_year is None:
			raise ValueError("Either `sol_dec` or `day_of_year` must be provided")
		else:
			sol_dec = solar_declination(day_of_year)
	else:
		_check_solar_declination_radians(sol_dec)

	if sha is None:
		sha = sunset_hour_angle(latitude, sol_dec)
	else:
		_check_sunset_hour_angle_radians(sha)

	tmp1 = SOLAR_CONSTANT / np.pi
	tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
	tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
	return tmp1 * ird * (tmp2 + tmp3)


def inverse_relative_distance_earth_sun(day_of_year=None):
	"""
	Calculate the inverse relative distance between earth and sun from day of the year.
	Based on FAO equation 23 in Allen et al (1998).

	Args:
		day_of_year: Day of the year [1 to 366]

	Returns:
		Inverse relative distance between earth and the sun
	"""
	_check_day_of_year(day_of_year)
	return 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year))
