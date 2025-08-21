"""
Module containing functionality to estimate reference evapotransporation (ETo), sometimes referred to as
potential evapotranspiration (PET), for a grass reference crop using the FAO-56 Penman-Monteith equation.
"""

import numpy as np
import pandas as pd
import xarray as xr


def fao56_penman_monteith(
	data: xr.Dataset | pd.DataFrame = None,
	net_rad: str | np.ndarray = "net_rad",
	t: str | np.ndarray = "2t",
	ws: str | np.ndarray = "ws",
	svp: str | np.ndarray = "svp",
	avp: str | np.ndarray = "avp",
	delta_svp: str | np.ndarray = "delta_svp",
	psy: str | np.ndarray = "psy",
	shf: str | np.ndarray = None,
) -> np.ndarray | pd.Series | xr.DataArray:
	"""
	Estimate reference evapotranspiration (ETo) from a hypothetical short grass reference surface using
	the FAO-56 Penman-Monteith equation. Based on equation 6 in Allen et al (1998).

	If `data==None`, the other arguments should all be `np.array` for each parameter. If `data` is
	a `pd.DataFrame` or `xr.Dataset`, the other arguments should be `str` of the names of each parameter
	in `data`.

	This is the method recommended by the Food and Agriculture Organisation of the United Nations (FAO)
	for estimating (ETo) for a short grass crop using limited meteorological data (see Allen et al, 1998).
	The FAO-56 Penman-Monteith equation requires site location, air temperature, humidity, radiation and
	wind speed data for daily, weekly, ten-day or monthly ETo calculations. It is important to verify the
	units of all input data.

	Args:
		data: pd.DataFrame or xr.Dataset containing columns/variables of all necessary parameters. If `None`,
		all other arguments must be of type `np.array` with values for each parameter.
		net_rad: Net radiation at crop surface [MJ m-2 day-1]. If necessary this can be estimated
		using ``solar_radiation.net_rad()``.
		t: Mean daily air temperature at 2 m height [deg Celcius].
		ws: Wind speed at 2 m height [m s-1]. If not measured at 2m, convert using ``wind_speed_at_2m()``.
		svp: Saturation vapour pressure [kPa]. Can be estimated using ``saturation_vapor_pressure_from_temperature()''.
		avp: Actual vapour pressure [kPa]. Can be estimated using a range of functions, in order of preference:
			1 - If dewpoint temperature data are available use ``actual_vapour_pressure_from_tdew()``.
			2 - If dry and wet bulb temperatures are available from a psychrometer
			use ``actual_vapour_pressure_from_twet_tdry()``.
			3 - If reliable minimum and maximum relative humidity data available
			use ``actual_vapour_pressure_from_rhmin_rhmax()``.
			4 - If measurement errors of relative humidity are large then use only maximum relative humidity
			using ``actual_vapour_pressure_from_rhmax()``.
			5 - If minimum and maximum relative humidity are not available but mean relative humidity is available
			then use ``actual_vapour_pressure_from_rhmean()`` (but this is less reliable than options 3 or 4).
			6 - If no data for the above are available then use ``actual_vapour_pressure_from_tmin()``.
			This function is less reliable in arid areas where it is recommended that 2 degrees Celsius is
			subtracted from the minimum temperature before it is passed to the function (following advice
			given in Annex 6 of Allen et al (1998).
		delta_svp: Slope of saturation vapour pressure curve [kPa degC-1]. Can be
		estimated using ``delta_saturation_vapor_pressure()``.
		psy: Psychrometric constant [kPa degC-1]. Can be estimatred using, in order of preference:
		``psy_const_of_psychrometer()`` or ``psy_const()``.
		shf: Soil heat flux (G) [MJ m-2 day-1] (default is 0.0, which is reasonable for a daily or 10-day time
		steps). For monthly time steps ``shf`` can be estimated using ``monthly_soil_heat_flux()`` or
		``monthly_soil_heat_flux2()``.

	Returns:
		Reference evapotranspiration (ETo) from a hypothetical grass reference surface [mm day-1].
		If `data` is `xr.Dataset`, returns a `xr.DataArray`. If `data` is `pd.DataFrame`, returns
		a `pd.Series`. Otherwise, return a `np.ndarray`.
	"""
	# If using numpy arrays
	if data is None:
		return _fao56_penman_monteith(net_rad, t, ws, svp, avp, delta_svp, psy, shf)

	dim_order = ["lat", "lon", "time"]

	# Convert xr.Dataset to pd.Dataframe
	if isinstance(data, xr.Dataset):
		df = data.to_dataframe(dim_order=dim_order)
		df.index.names = dim_order
	elif isinstance(data, pd.DataFrame):
		df = data.reorder_levels(dim_order, axis=0) if all([i in data.index.names for i in dim_order]) else data

	# Check all required parameters are given
	for name, argument_name in zip(
		["net_rad", "t", "ws", "svp", "avp", "delta_svp", "psy"],
		[net_rad, t, ws, svp, avp, delta_svp, psy],
		strict=False,
	):
		if argument_name not in df.columns:
			raise ValueError(f"Variable name {name}={argument_name} not found in `data`")

	# initialize value of soil heat flux variable
	if shf is not None:
		if shf in df.columns:
			shf = df[shf]
		else:
			raise ValueError(f"Variable name `shf`={shf} not found in `data`")

	# eto is a pd.Series
	eto = _fao56_penman_monteith(df[net_rad], df[t], df[ws], df[svp], df[avp], df[delta_svp], df[psy], shf)
	eto.name = "eto"

	if isinstance(data, xr.Dataset):
		eto = eto.to_xarray()
		eto = eto.assign_attrs(
			{
				"units": "mm day-1",
				"description": "ETo reference evapotranspiration [mm day-1]",
			}
		)
		eto = eto.to_dataset()
	return eto


def _fao56_penman_monteith(net_rad, t, ws, svp, avp, delta_svp, psy, shf=0.0):
	"""
	Estimate reference evapotranspiration (ETo) from a hypothetical short grass reference surface using
	the FAO-56 Penman-Monteith equation. Based on equation 6 in Allen et al (1998).

	This is the method recommended by the Food and Agriculture Organisation of the United Nations (FAO)
	for estimating (ETo) for a short grass crop using limited meteorological data (see Allen et al, 1998).
	The FAO-56 Penman-Monteith equation requires site location, air temperature, humidity, radiation and
	wind speed data for daily, weekly, ten-day or monthly ETo calculations. It is important to verify the
	units of all input data.

	Args:
		net_rad: Net radiation at crop surface [MJ m-2 day-1]. If necessary this can be estimated
		using ``solar_radiation.net_rad()``.
		t: Mean daily air temperature at 2 m height [deg Celcius].
		ws: Wind speed at 2 m height [m s-1]. If not measured at 2m, convert using ``wind_speed_at_2m()``.
		svp: Saturation vapour pressure [kPa]. Can be estimated using ``saturation_vapor_pressure_from_temperature()''.
		avp: Actual vapour pressure [kPa]. Can be estimated using a range of functions, in order of preference:
			1 - If dewpoint temperature data are available use ``actual_vapour_pressure_from_tdew()``.
			2 - If dry and wet bulb temperatures are available from a psychrometer
			use ``actual_vapour_pressure_from_twet_tdry()``.
			3 - If reliable minimum and maximum relative humidity data available
			use ``actual_vapour_pressure_from_rhmin_rhmax()``.
			4 - If measurement errors of relative humidity are large then use only maximum relative humidity
			using ``actual_vapour_pressure_from_rhmax()``.
			5 - If minimum and maximum relative humidity are not available but mean relative humidity is available
			then use ``actual_vapour_pressure_from_rhmean()`` (but this is less reliable than options 3 or 4).
			6 - If no data for the above are available then use ``actual_vapour_pressure_from_tmin()``.
			This function is less reliable in arid areas where it is recommended that 2 degrees Celsius is
			subtracted from the minimum temperature before it is passed to the function (following advice
			given in Annex 6 of Allen et al (1998).
		delta_svp: Slope of saturation vapour pressure curve [kPa degC-1]. Can be
		estimated using ``delta_saturation_vapor_pressure()``.
		psy: Psychrometric constant [kPa degC-1]. Can be estimatred using, in order of preference:
		``psy_const_of_psychrometer()`` or ``psy_const()``.
		shf: Soil heat flux (G) [MJ m-2 day-1] (default is 0.0, which is reasonable for a daily or 10-day time
		steps). For monthly time steps ``shf`` can be estimated using ``monthly_soil_heat_flux()`` or
		``monthly_soil_heat_flux2()``.

	Returns:
		Reference evapotranspiration (ETo) from a hypothetical grass reference surface [mm day-1].
	"""
	if shf is None:
		shf = 0.0

	# Check all required parameters are given
	for name, variable in zip(
		["net_rad", "t", "ws", "svp", "avp", "delta_svp", "psy", "shf"],
		[net_rad, t, ws, svp, avp, delta_svp, psy, shf],
		strict=False,
	):
		if variable is None:
			raise ValueError(f"Parameter {name} is not given")

	numerator = (0.408 * (net_rad - shf) * delta_svp) + ((psy * 891.3 * ws * (svp - avp)) / (t + 273))
	denominator = delta_svp + (psy * (1 + 0.3365 * ws))
	return numerator / denominator
