"""
SPI: Number of rolling months depends on the timescale of interest, where 3 month SPI
is used for a short-term or seasonal drought index, 12 month SPI for an
intermediate-term drought index, and 48 month SPI for a long-term drought index.

TODO:
- An addition parameter should be the output timestep, as e.g. input data could be daily, but we
want data with monthly timesteps. A resampling operation should be carried out PRIOR to the rolling operation.
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import xarray as xr

import terrapyn as tp

# TODO - this could be implemented in fit Gamma PDF if there is also the option to set a min_threshold
# just above zero, instead of equal to zero, e.g.
# prob_zero = _fraction_less_equal_threshold(values, 0.0001)

# def _fraction_less_equal_threshold(values: np.ndarray = None, threshold: float = 0) -> float:
#     return np.sum(values <= threshold) / values.shape[0]


# def _ensure_monthly_timesteps(data):
#     times = tp.time.get_time_from_data(data, time_dim=time_dim)
#     inferred_freq = pd.infer_freq(times)
#     # return inferred_freq[0] == 'M'

#     # Check if frequency of data is higher than monthly
#     if tp.time.FREQUENCY_DICT[inferred_freq[0]] < tp.time.FREQUENCY_DICT['M']:

# # Resample to monthly

#     inferred_freq


# def _infer_frequency(data, time_dim: str = "time"):
#     """Return the frequency of data as a np.timedelta64"""
#     times = tp.time.get_time_from_data(data, time_dim=time_dim)
#     return np.diff(timests)).mean()


def _fit_gamma_pdf(array: np.ndarray = None) -> np.ndarray:
	"""
	Fits a Gamma PDF to a 1-D array and returns the shape and scale parameters,
	where shape = alpha and scale = 1 / beta see
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html.

	Values < 0 are automatically clipped to 0, and only finite values > 0 are included in the fit.
	If only non-finite values are given, a parameter array of NaN is returned.

	Args:
		array: Array of values to fit

	Returns:
		Array of [shape, scale] parameters for fitted Gamma PDF
	"""
	# Include only finite values > 0
	finite_values_mask = np.isfinite(array) & (array > 0)

	if np.any(finite_values_mask):
		finite_values = array[finite_values_mask]

		# # Fit Gamma PDF to data using scipy
		# shape, _, scale = st.gamma.fit(finite_values)
		# return np.array([shape, scale])

		# Fit Gamma PDF using approximation - see Lloyd-Hughes and Saunders 2002, A drought climatology for Europe
		log_values = np.log(finite_values)
		mean_values = np.mean(finite_values)
		A = np.log(mean_values) - np.mean(log_values)
		alpha = 1 / (4 * A) * (1 + np.sqrt(1 + 4 / 3 * A))
		beta = mean_values / alpha
		return np.array([alpha, beta])

	else:
		return np.array([np.nan, np.nan])


def _fit_gamma_pdf_dataarray(da: xr.DataArray = None, time_dim: str = "time") -> xr.Dataset:
	"""
	Fits a Gamma PDF to an xr.DataArray and returns an xr.Dataset with the Gamma function shape and scale parameters,
	where shape = alpha and scale = 1 / beta see
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html.

	Values < 0 are automatically clipped to 0, and only finite values are included in the fit.

	Args:
		da: xr.DataArray of values to fit, with coordinates of latitude, longitude, and time.
		time_dim: Name of the time coordinate in `da`.

	Returns:
		xr.Dataset of the Gamma function shape and scale parameters
	"""
	# Fit the Gamma function to all values in `time_dim`, for each coordinate
	parameters = xr.apply_ufunc(
		_fit_gamma_pdf,
		da,
		input_core_dims=[[time_dim]],
		output_core_dims=[["parameter"]],
		vectorize=True,
		dask="parallelized",
		output_dtypes=["float32"],
		dask_gufunc_kwargs=dict(output_sizes={"parameter": 2}),
	)

	# Convert dataarray to dataset with labelled variables
	parameters = parameters.to_dataset(dim="parameter").rename({0: "shape", 1: "scale"})
	return parameters


def _calc_gamma_cdf(
	array: pd.Series | np.ndarray = None, shape: float = None, scale: float = None
) -> pd.Series | np.ndarray:
	"""
	Returns the Gamma CDF for values in a 1-D array, for the given shape and scale parameters,
	where shape = alpha and scale = 1 / beta
	see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

	Scales the Gamma CDF by the fraction of zero values in the data - see "A drought climatology
	for Europe, B. Lloyd-Hughes and M.A. Saunders 2002, https://doi.org/10.1002/joc.846

	Args:
		array: Data values to use for generating the CDF.
		shape: The shape (alpha) parameter for the Gamma function.
		scale: The scale (1 / beta) parameter for the Gamma function.

	Returns:
		Array of values for the Gamma function CDF
	"""
	# Probabity of zeros
	# prob_zero = _fraction_less_equal_threshold(array, min_threshold)
	prob_zero = np.sum(array == 0.0) / array.shape[0]

	# Calculate the gamma CDF of the data, fixing location to zero
	gamma_cdf = prob_zero + (1 - prob_zero) * st.gamma.cdf(array, a=shape, loc=0, scale=scale)

	if isinstance(array, pd.Series):
		return pd.Series(gamma_cdf, index=array.index)
	else:
		return gamma_cdf


def _calc_gamma_cdf_dataarray(
	da_values: xr.DataArray = None,
	da_parameters: xr.Dataset = None,
	time_dim: str = "time",
	shape_dim: str = "shape",
	scale_dim: str = "scale",
) -> xr.DataArray:
	"""
	Calculates a Gamma CDF for valus at each coordinate in a xr.DataArray using the shape and scale parameters,
	where shape = alpha and scale = 1 / beta see
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html.

	Args:
		da_values: xr.DataArray of values, with coordinates of latitude, longitude, and time.
		da_parameters: xr.Dataset of Gamma function parameters "shape" and "scale"
		time_dim: Name of the time coordinate in `da_values`.
		shape_dim: Name of the Gamma "shape" parameter in `da_parameters`.
		scale_dim: Name of the Gamma "scale" parameter in `da_parameters`.

	Returns:
		xr.DataArray for the Gamma function CDF for the given values, shape and scale parameters.
	"""
	# Fit the Gamma function to all values in `time_dim`, for each coordinate
	return xr.apply_ufunc(
		_calc_gamma_cdf,
		da_values,
		da_parameters[shape_dim],
		da_parameters[scale_dim],
		input_core_dims=[[time_dim], [], []],
		output_core_dims=[[time_dim]],
		vectorize=True,
		dask="parallelized",
		output_dtypes=["float32"],
	)


def fit_gamma_pdf(data: pd.Series | xr.DataArray = None, time_dim: str = "time") -> tuple | xr.DataArray:
	"""
	Fits a Gamma PDF to data and return the shape and scale parameters of the Gamma function,
	where shape = alpha and scale = 1 / beta see
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html.

	Normally data is a rolling sum for monthly data.

	Accept pd.Series or xr.DataArray, where for pd.Series a tuple of (shape, scale) is returned, and
	for xr.DataArray a xr.Dataset is returned with shape and scale as variables.

	Values < 0 are automatically clipped to 0, and only finite values are included in the fit.

	Args:
		data: Data to fit
		time_dim: (Optional) Name of the time coordinate in `data`. Only applies if using a xr.DataArray.

	Returns:
		Shape and scale parameters for the fitted Gamma PDF
	"""
	if not isinstance(data, pd.Series | xr.DataArray):
		raise TypeError("data must be of type pd.Series or xr.DataArray")

	# Set negative values to zero
	data = data.clip(0, None)

	if isinstance(data, pd.Series):
		return _fit_gamma_pdf(data)

	else:
		if tp.dask_utils.uses_dask(data):
			# Re-chunk along time_dim if using Dask
			data = tp.dask_utils.chunk_xarray(data, coords_no_chunking=time_dim)

		return _fit_gamma_pdf_dataarray(da=data, time_dim=time_dim)


def calc_gamma_cdf(
	data: pd.Series | xr.DataArray = None,
	gamma_parameters: tuple | xr.Dataset = None,
	time_dim: str = "time",
	shape_dim: str = "shape",
	scale_dim: str = "scale",
) -> tuple | xr.DataArray:
	"""
	Calculates a Gamma CDF for valus at each coordinate in a pd.Series or xr.DataArray using
	the shape and scale parameters, where shape = alpha and scale = 1 / beta see
	https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html.

	Args:
		data: pd.Series or xr.DataArray of values. xr.DataArray should have coordinates of latitude,
		longitude, and time (`time_dim').
		gamma_parameters: Gamma function parameters, either a Tuple of ("shape", "scale") or xr.Dataset
		with variables of "shape", "scale".
		time_dim: Name of the time coordinate in `gamma_parameters`. Only applies to xr.Dataset.
		shape_dim: Name of the Gamma "shape" parameter in `gamma_parameters`. Only applies to xr.Dataset.
		scale_dim: Name of the Gamma "scale" parameter in `gamma_parameters`. Only applies to xr.Dataset.

	Returns:
		Parameters of the Gamma function CDF for the given values (shape and scale parameters).
	"""
	if not isinstance(data, pd.Series | xr.DataArray):
		raise TypeError("data must be of type pd.Series or xr.DataArray")

	if isinstance(data, pd.Series):
		return _calc_gamma_cdf(data, shape=gamma_parameters[0], scale=gamma_parameters[1])

	else:
		if tp.dask_utils.uses_dask(data):
			# Re-chunk along time_dim if using Dask
			data = tp.dask_utils.chunk_xarray(data, coords_no_chunking=time_dim)

		return _calc_gamma_cdf_dataarray(
			data, gamma_parameters, time_dim=time_dim, shape_dim=shape_dim, scale_dim=scale_dim
		)


def _cdf_to_normal_ppf(cdf: np.ndarray = None) -> np.ndarray:
	"""Apply the inverse normal distribution to a CDF to yield a normal
	Percent point function (PPF) with mean = 0 and std = 1
	"""
	normal_ppf = st.norm.ppf(cdf, loc=0, scale=1)
	normal_ppf[np.isinf(normal_ppf)] = np.nan
	return normal_ppf


def _cdf_to_normal_ppf_dataarray(da: xr.DataArray = None, time_dim: str = "time") -> xr.DataArray:
	"""Apply the inverse normal distribution to a CDF to yield a normal PPF with mean = 0 and std = 1"""
	return xr.apply_ufunc(
		_cdf_to_normal_ppf,
		da,
		input_core_dims=[[time_dim]],
		output_core_dims=[[time_dim]],
		vectorize=True,
		dask="parallelized",
		output_dtypes=["float32"],
	)


def cdf_to_normal_ppf(data: pd.Series | xr.DataArray = None, time_dim: str = "time") -> pd.Series | xr.DataArray:
	"""
	Apply the inverse normal distribution to a CDF to yield a normal PPF with mean = 0 and std = 1

	Args:
		data: pd.Series or xr.DataArray of values. xr.DataArray should have coordinates of latitude,
		longitude, and time (`time_dim').
		time_dim: Name of the time coordinate in `data`. Only applies to xr.DataArray.

	Returns:
		The computed values of the inverse normal distribution for the given CDF
	"""
	if not isinstance(data, pd.Series | xr.DataArray):
		raise TypeError("data must be of type pd.Series or xr.DataArray")

	if isinstance(data, pd.Series):
		normal_pdf = _cdf_to_normal_ppf(data)
		return pd.Series(normal_pdf, index=data.index)

	else:
		if tp.dask_utils.uses_dask(data):
			# Re-chunk along time_dim if using Dask
			data = tp.dask_utils.chunk_xarray(data, coords_no_chunking=time_dim)
		return _cdf_to_normal_ppf_dataarray(data, time_dim=time_dim)


def calc_spi(
	data: pd.Series | xr.DataArray = None,
	n_months: int = 3,
	gamma_parameters: pd.Series | xr.DataArray = None,
	return_gamma_params: bool = False,
	time_dim: str = "time",
) -> pd.Series | xr.DataArray:
	"""Calculate SPI, where `n_months` is the timescale of interest: 3 month SPI is used for a
	short-term or seasonal drought index, 12 month SPI for an intermediate-term drought index,
	and 48 month SPI for a long-term drought index.

	Groups of monthly data are modelled using a Gamma Distritibution, and the corresponding
	CDF of model is transformed to a Normal PPF for each monthly value, yielding the SPI value.

	Args:
		data: Data, with values on a monthly timestep with datetimes as `time_dim` index
		n_months: The number of months over which to calculate the rolling mean (the timescale of interest)
		gamma_parameters: (Optional) The shape and scale parameters the define a Gamma distribution. If given,
		these parameters are used instead of fitting the data. For `pd.Series` the index must be the month,
		with values of a Tuple of (shape, scale).
		return_gamma_params: If `True` then return fitted Gamma Parameters.
		time_dim: Name of the time coordinate in `data`. Only applies to xr.DataArray.

	Returns:
		SPI values for each month
	"""
	# Calculate rolling mean over `n_months` with minimum of `n_months` in the window
	data_rolling = tp.time.rolling(data, n_periods=n_months, min_periods=n_months, method="mean", time_dim=time_dim)

	# Group rolling data by month
	data_rolling_grouped = tp.time.groupby_time(data_rolling, grouping="month")

	# Use Gamma Parameters provided, otherwise fit the Gamma distribution
	if gamma_parameters is None:
		# Calculate Gamma distribution parameters for each monthly group
		gamma_parameters = data_rolling_grouped.apply(tp.indices.spi.fit_gamma_pdf)

	# For each month, calculate the Gamma CDF
	if isinstance(data, pd.Series):
		# Check length of gamma_parameters is equal to the number of months
		# gamma_parameters is a list of tuples of (shape, scale)
		# if len(gamma_parameters) != data_rolling_grouped.ngroups:
		#     raise ValueError(f"`gamma_parameters` has length {len(gamma_parameters)}")

		gamma_cdf_list = [
			tp.indices.spi.calc_gamma_cdf(group, gamma_parameters.loc[label]) for label, group in data_rolling_grouped
		]

		# For each month, transform the Gamma CDF to a Normal PPF, then sort the data by time
		normal_ppf = pd.concat([tp.indices.spi.cdf_to_normal_ppf(item) for item in gamma_cdf_list]).sort_index()

	else:
		gamma_cdf_list = [
			tp.indices.spi.calc_gamma_cdf(group, gamma_parameters.sel({"month": label}).drop("month"))
			for label, group in data_rolling_grouped
		]

		# For each month, transform the Gamma CDF to a Normal PPF, then sort the data by time
		normal_ppf = xr.concat([tp.indices.spi.cdf_to_normal_ppf(item) for item in gamma_cdf_list], dim="time").sortby(
			"time"
		)

	if return_gamma_params:
		return normal_ppf, gamma_parameters
	else:
		return normal_ppf
