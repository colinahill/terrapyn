import typing as T

import numpy as np
import pandas as pd
import xarray as xr

from terrapyn.utils import ensure_list

VALUES_AND_ARRAY_LIKE = float | int | np.ndarray | list | pd.Series | pd.DataFrame


def error(x: VALUES_AND_ARRAY_LIKE, y: VALUES_AND_ARRAY_LIKE) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate the error between two values or arrays, where `error = x - y`.

	Args:
		x: First array/value
		y: Second array/value

	Returns:
		The error, in the same format as the input arrays
	"""
	if isinstance(x, pd.Series | pd.DataFrame):
		if isinstance(y, pd.Series | pd.DataFrame):
			y = y.to_numpy()
		return x.subtract(y, axis=0)
	else:
		return x - y


def r2(model: VALUES_AND_ARRAY_LIKE = None, observations: VALUES_AND_ARRAY_LIKE = None) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate R^2 between two values/arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values

	Returns:
		R^2
	"""
	return np.corrcoef(model, observations)[0, 1] ** 2


def normalized_root_mean_squared_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Normalized Root Mean Squared Error between two values/arrays.
	"""
	return root_mean_squared_error(model, observations, axis=axis) / np.nanmean(observations, axis=axis)


def root_mean_squared_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Root Mean Squared Error between two values/arrays. Axis can be selected for multi-dimensional arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values
		axis: Axis over which to perform the calculation, where 0=columns, 1=rows, None=all elements

	Returns:
		Root Mean squared error
	"""
	return np.sqrt(mean_squared_error(model, observations, axis=axis))


def normalized_mean_squared_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Normalized Mean Squared Error between two values/arrays.
	"""
	return mean_squared_error(model, observations, axis=axis) / np.nanmean(observations, axis=axis)


def mean_squared_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Mean Squared Error between two values/arrays. Axis can be selected for multi-dimensional arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values
		axis: Axis over which to perform the calculation, where 0=columns, 1=rows, None=all elements

	Returns:
		Mean squared error
	"""
	return np.nanmean(np.square(error(model, observations)), axis=axis)


def mean_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Mean Error between two values/arrays. Axis can be selected for multi-dimensional arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values
		axis: Axis over which to perform the calculation

	Returns:
		Mean Error
	"""
	n, error = _check_dims_and_compute_error(model, observations, axis)
	if error is not None:
		return np.sum(error, axis=axis) / n


def normalized_mean_absolute_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Normalized Mean Absolute Error between two values/arrays.
	"""
	return mean_absolute_error(model, observations, axis=axis) / np.nanmean(observations, axis=axis)


def mean_absolute_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Mean Absolute Error between two values/arrays. Axis can be selected for multi-dimensional arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values
		axis: Axis over which to perform the calculation, where 0=columns, 1=rows, None=all elements

	Returns:
		Mean Absolute Error
	"""
	n, error = _check_dims_and_compute_error(model, observations, axis)
	if error is not None:
		return np.sum(np.abs(error), axis=axis) / n


def bias(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Bias between two values/arrays, excluding nan values in both arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values
		axis: Axis over which to perform the calculation, where 0=columns, 1=rows, None=all elements

	Returns:
		Bias
	"""
	model_finite, obs_finite = _get_matching_finite_values(model, observations)
	return np.divide(np.sum(model_finite), np.sum(obs_finite))


def efficiency(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Calculate Efficiency score between two values/arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values

	Returns:
		Efficiency score
	"""
	model_finite, observations_finite = _get_matching_finite_values(model, observations)
	numerator = np.sum(np.square(error(model_finite, observations_finite)))
	denominator = np.sum(np.square(error(observations_finite, np.mean(observations_finite))))
	return 1.0 - np.divide(numerator, denominator)


def skill_score(reference, test):
	"""
	Calculate the Skill Score between two score values. Typically used with mean squared error or mean absolute error to
	compare score of observations-to-climatology with observations-to-model.

	Args:
		reference: Score of reference (e.g. MSE of observations compared to climatology)
		test: Score of test variable (e.g. MSE of model compared to observations)

	Returns:
		Skill Score
	"""
	return 1.0 - test / reference


def normalized_mae_dataarray(
	model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time"
) -> xr.DataArray:
	"""
	Calculate Normalized Mean Absolute Error between two xr.DataArrays
	"""
	return mae_dataarray(model, observations, dim) / observations.mean(dim=dim)


def mae_dataarray(model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time") -> xr.DataArray:
	"""
	Calculate Mean Absolute Error between two xr.DataArrays

	Args:
		model: Forecast/Satellite/Model dataarray
		observations: Observations dataarray
		dim: Dimension of dataarray over which to compute the score

	Returns:
		Mean Absolute Error
	"""
	return np.abs(model - observations).mean(dim=dim)


def r2_dataset(ds, model: str = None, observations: str = None, dim: str = None) -> xr.Dataset | np.float64:
	"""
	Calculate R^2 between variables in an xr.Dataset
	"""
	if dim is None:
		return r2(ds[model].to_numpy().flatten(), ds[observations].to_numpy().flatten())
	else:
		return xr.apply_ufunc(
			r2,
			ds[model],
			ds[observations],
			input_core_dims=[[dim], [dim]],
			vectorize=True,
			dask="parallelized",
			output_dtypes=[float],
		)


def bias_dataarray(model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time") -> xr.DataArray:
	"""
	Calculate Bias between two xr.DataArrays, excluding nan values.

	Args:
		model: Forecast/Satellite/Model dataarray
		observations: Observations dataarray
		dim: Dimension of dataarray over which to compute the score

	Returns:
		Bias
	"""
	model_values = model.to_numpy()
	obs_values = observations.to_numpy()
	return bias(model_values, obs_values)


def normalized_mse_dataarray(
	model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time"
) -> xr.DataArray:
	"""
	Calculate Normalized Mean Squared Error between two xr.DataArrays
	"""
	return mse_dataarray(model, observations, dim) / observations.mean(dim=dim)


def mse_dataarray(model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time") -> xr.DataArray:
	"""
	Calculate Mean Squared Error between two xr.DataArrays

	Args:
		model: Forecast/Satellite/Model dataarray
		observations: Observations dataarray
		dim: Dimension of dataarray over which to compute the score

	Returns:
		Mean Squared Error
	"""
	return np.square(model - observations).mean(dim=dim)


def normalized_rmse_dataarray(
	model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time"
) -> xr.DataArray:
	"""
	Calculate Normalized Root Mean Squared Error between two xr.DataArrays
	"""
	return rmse_dataarray(model, observations, dim) / observations.mean(dim=dim)


def rmse_dataarray(model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time") -> xr.DataArray:
	"""
	Calculate Root Mean Squared Error between two xr.Dataset/Arrays.

	Args:
		model: Forecast/Satellite/Model dataarray
		observations: Observations dataarray
		dim: Dimension of dataarray over which to compute the score

	Returns:
		Root Mean squared error
	"""
	return np.sqrt(mse_dataarray(model, observations, dim))


def me_dataarray(model: xr.DataArray = None, observations: xr.DataArray = None, dim: str = "time") -> xr.DataArray:
	"""
	Calculate Mean Error between two xr.Dataset/Arrays.

	Args:
		model: Forecast/Satellite/Model dataarray
		observations: Observations dataarray
		dim: Dimension of dataarray over which to compute the score

	Returns:
		Mean Error
	"""
	return (model - observations).mean(dim=dim)


def error_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	**kwargs,
) -> pd.DataFrame:
	"""
	Calc ERROR for columns in a pandas dataframe. If multiple values of `model_name` and/or
	`obs_name` are given (as a `list`), the scores are computed as follows:
	- Equal lengths of `model_name` and `obs_name`: results in the pair-wise scores
	(column 1 with 1, column 2 with 2 etc.).
	- A list of columns in `model_name` and fewer columns in `obs_name`: Scores are computed as all 'model' columns
	with the first 'obs' column, then all 'model' columns with the second 'obs' column etc.

	Args:
		df: the input dataframe
		model_name: Name(s) of columns for the prediction/model
		obs_name: Name(s) of columns for the truth/observations
		output_index_names: Names to assign to the columns of the output dataframe

	Returns:
		Error
	"""
	dims_equal, n_models, n_obs = _check_dims_compatible_with_length_of_list(model_name, obs_name, output_index_names)

	model_name = ensure_list(model_name)
	obs_name = ensure_list(obs_name)

	if output_index_names is None:
		generate_index_names = True
	else:
		generate_index_names = False
		output_index_names = ensure_list(output_index_names)

	if dims_equal:
		score = error(df[model_name], df[obs_name])
		if generate_index_names:
			output_index_names = [f"{i}_{j}" for i, j in zip(model_name, obs_name, strict=False)]
		score.columns = output_index_names
	else:
		# Multiple models/predictions and fewer truth/observation columns. Scores are computed
		# between all models and the first observation column, then all models and the second observation column etc.
		df_list = []
		for col in obs_name:
			score = error(df[model_name], df[col])
			if generate_index_names:
				output_index_names = [f"{i}_{col}" for i in model_name]
				score.columns = output_index_names
			df_list.append(score)
		score = pd.concat(df_list, axis=1)
		if not generate_index_names:
			score.columns = output_index_names

	if df.index.name is not None:
		score.index.name = df.index.name
	else:
		score.index.names = df.index.names
	return score


def normalized_mae_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
) -> pd.DataFrame:
	"""
	Calculate Normalized Mean Absolute Error between two values/arrays.
	"""
	score = mae_df(df, model_name, obs_name, output_index_names, axis)
	score = _divide_rowwise(score, df[obs_name].mean(axis=axis))
	score.name = "nmae"
	return score


def mae_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
) -> pd.DataFrame:
	"""
	Calc MAE for columns in a pandas dataframe

	Args:
		df: the input dataframe
		model_name: Name(s) of columns for the prediction/model
		obs_name: Name(s) of columns for the truth/observations
		output_index_names: Names to assign to the index/columns of the output dataframe

	Returns:
		MAE
	"""
	error_score = error_df(df, model_name, obs_name, output_index_names)
	score = error_score.abs().mean(axis=axis)
	if output_index_names is not None:
		output_index_names = ensure_list(output_index_names)
		score.index = output_index_names
	score.name = "mae"
	return score


def normalized_mse_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
) -> pd.DataFrame:
	"""
	Calculate Normalized Mean Squared Error between two values/arrays.
	"""
	score = mse_df(df, model_name, obs_name, output_index_names, axis)
	score = _divide_rowwise(score, df[obs_name].mean(axis=axis))
	score.name = "nmse"
	return score


def mse_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
) -> pd.DataFrame:
	"""
	Calc MSE for columns in a pandas dataframe

	Args:
		df: the input dataframe
		model_name: Name(s) of columns for the prediction/model
		obs_name: Name(s) of columns for the truth/observations
		output_index_names: Names to assign to the index/columns of the output dataframe

	Returns:
		MSE
	"""
	error_score = error_df(df, model_name, obs_name, output_index_names)
	score = error_score.pow(2).mean(axis=axis)
	if output_index_names is not None:
		output_index_names = ensure_list(output_index_names)
		score.index = output_index_names
	score.name = "mse"
	return score


def me_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
) -> pd.DataFrame:
	"""
	Calc ME for columns in a pandas dataframe

	Args:
		df: the input dataframe
		model_name: Name(s) of columns for the prediction/model
		obs_name: Name(s) of columns for the truth/observations
		output_index_names: Names to assign to the index/columns of the output dataframe

	Returns:
		ME
	"""
	error_score = error_df(df, model_name, obs_name, output_index_names)
	score = error_score.mean(axis=axis)
	if output_index_names is not None:
		output_index_names = ensure_list(output_index_names)
		score.index = output_index_names
	score.name = "me"
	return score


def _divide_rowwise(data, divisor):
	if isinstance(divisor, pd.Series | pd.DataFrame):
		return data.div(divisor.to_numpy(), axis=0)
	elif isinstance(divisor, np.ndarray | float | int):
		return data.div(divisor, axis=0)


def normalized_rmse_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
) -> pd.DataFrame:
	"""
	Calculate Normalized Root Mean Squared Error between two values/arrays.
	"""
	score = rmse_df(df, model_name, obs_name, output_index_names, axis)
	score = _divide_rowwise(score, df[obs_name].mean(axis=axis))
	score.name = "nrmse"
	return score


def rmse_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
) -> pd.DataFrame:
	"""
	Calc RMSE for columns in a pandas dataframe

	Args:
		df: the input dataframe
		model_name: Name(s) of columns for the prediction/model
		obs_name: Name(s) of columns for the truth/observations
		output_index_names: Names to assign to the index/columns of the output dataframe

	Returns:
		RMSE
	"""
	mse_score = mse_df(df, model_name, obs_name, output_index_names, axis)
	score = np.power(mse_score, 0.5)
	score.name = "rmse"
	return score


def bias_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
) -> pd.DataFrame:
	"""
	Calc Bias for columns in a pandas dataframe

	Args:
		df: the input dataframe
		model_name: Name(s) of columns for the prediction/model
		obs_name: Name(s) of columns for the truth/observations
		output_index_names: Names to assign to the index/columns of the output dataframe

	Returns:
		Bias score
	"""
	dims_equal, n_models, n_obs = _check_dims_compatible_with_length_of_list(model_name, obs_name, output_index_names)

	model_name = ensure_list(model_name)
	obs_name = ensure_list(obs_name)

	# sum of each column
	sum_models = df[model_name].sum(axis=0, skipna=True)
	sum_obs = df[obs_name].sum(axis=0, skipna=True)

	if dims_equal:
		score = sum_models / sum_obs.to_numpy()
		if output_index_names is None:
			output_index_names = [f"{i}_{j}" for i, j in zip(model_name, obs_name, strict=False)]
		score.index = output_index_names
	else:
		# Multiple models/predictions and fewer truth/observation columns. Scores are computed
		# between all models and the first observation column, then all models and the second observation column etc.
		df_list = []
		for col in obs_name:
			score = sum_models / sum_obs.loc[[col]].to_numpy()
			if output_index_names is None:
				output_index_names = [f"{i}_{col}" for i in model_name]
				score.index = output_index_names
			df_list.append(score)
		score = pd.concat(df_list, axis=0)
		if output_index_names is not None:
			output_index_names = ensure_list(output_index_names)
			score.index = output_index_names
	score.name = "bias"
	return score


def efficiency_df(
	df: pd.DataFrame = None,
	model_name: str | list[str] = None,
	obs_name: str | list[str] = None,
	output_index_names: T.Iterable = None,
) -> pd.DataFrame:
	"""
	Calc Efficiency score for columns in a pandas dataframe

	Args:
		df: the input dataframe
		model_name: Name(s) of columns for the prediction/model
		obs_name: Name(s) of columns for the truth/observations
		output_index_names: Names to assign to the index/columns of the output dataframe

	Returns:
		Efficiency score
	"""
	dims_equal, n_models, n_obs = _check_dims_compatible_with_length_of_list(model_name, obs_name, output_index_names)

	model_name = ensure_list(model_name)
	obs_name = ensure_list(obs_name)

	if dims_equal:
		# loop over pairs of columns
		score_list = []
		index = []
		for mod, obs in zip(model_name, obs_name, strict=False):
			score = efficiency(df[mod], df[obs])
			score_list.append(score)
			index.append(f"{mod}_{obs}")
		output_index_names = index if output_index_names is None else ensure_list(output_index_names)
		score = pd.Series(score_list, index=output_index_names, name="efficiency")
		return score
	else:
		# loop over models, for each obs
		score_list = []
		index = []
		for mod in model_name:
			for obs in obs_name:
				score = efficiency(df[mod], df[obs])
				score_list.append(score)
				index.append(f"{mod}_{obs}")
		output_index_names = index if output_index_names is None else ensure_list(output_index_names)
		score = pd.Series(score_list, index=output_index_names, name="efficiency")
		return score


def _check_dims_and_compute_error(
	model: VALUES_AND_ARRAY_LIKE = None,
	observations: VALUES_AND_ARRAY_LIKE = None,
	axis=None,
) -> VALUES_AND_ARRAY_LIKE:
	"""
	Check dimensions and calculate the error between two values/arrays.

	Args:
		model: Forecast/Satellite/Model values
		observations: Observations values
		axis: Axis over which to perform the calculation

	Returns:
		length of array, error array
	"""
	check, model_dims, obs_dims = _check_dimensions_are_equal(
		model, observations, calc_dims=True, return_dims=True, verbose=True
	)
	# Continue if dimensions are equal
	if check:
		# Length of array for computation
		if axis is not None:
			if model_dims == 1 and axis > 0:
				print("Values are 1-D. Reverting to axis=0")
				axis = 0
			n = model_dims
		else:
			n = model_dims if isinstance(model_dims, int) else model_dims[0]
		return n, error(model, observations)
	else:
		raise ValueError(f"Data dimensions are not equal, cannot compute error {model_dims} vs {obs_dims}")


def _get_matching_finite_values(x: T.Iterable, y: T.Iterable) -> tuple[T.Iterable, T.Iterable]:
	"""
	Find overlapping finite (non-NaN) values for two input arrays

	Args:
		x: First array
		y: Second array

	Returns:
		First and second arrays with finte values that have overlapping indexes
	"""
	mask_x = np.isfinite(x)
	mask_y = np.isfinite(y)
	mask = mask_x & mask_y
	return x[mask], y[mask]


def _get_dimensionality(values) -> tuple:
	"""Get dimensions of data. Returns 1 for single values, otherwise the shape"""
	if isinstance(values, str | float | int):
		return 1
	elif isinstance(values, pd.Series | pd.DataFrame | np.ndarray | list):
		shape = np.shape(values)
		if len(shape) == 1:
			return shape[0]
		else:
			return shape


def _check_dimensions_are_equal(
	a: VALUES_AND_ARRAY_LIKE,
	b: VALUES_AND_ARRAY_LIKE,
	calc_dims: bool = True,
	return_dims: bool = True,
	verbose: bool = False,
) -> (bool, tuple | None, tuple | None):
	"""
	Check that the dimensions of two arrays (or values) are equal.

	Returns True or False, and optionally the dimensions of the input arrays.

	Args:
		a: First array/value
		b: Second array/value
		calc_dims: If True, get the dimensionality of the input arrays. If False, the, input arrays must be dimensions.
		return_dims: If True, additionally return the dimensionality of the input arrays.
	Returns:
		True/False if dimensions are equal, and optionally, T.Tuples of the dimensions.
	"""
	if calc_dims:
		a_dims = _get_dimensionality(a)
		b_dims = _get_dimensionality(b)
	else:
		a_dims = a
		b_dims = b

	if a_dims == b_dims:
		if return_dims:
			return True, a_dims, b_dims
		else:
			return True
	else:
		if verbose:
			print(f"Error: Dimensions of values/arrays are not equal - {a_dims} vs {b_dims}")

		if return_dims:
			return False, a_dims, b_dims
		else:
			return False


def _check_dims_compatible_with_length_of_list(model_list, obs_list, output_index_names: list = None):
	dims_equal, n_models, n_obs = _check_dimensions_are_equal(model_list, obs_list, calc_dims=True, return_dims=True)
	if not dims_equal and output_index_names is not None and len(output_index_names) != n_models * n_obs:
		raise ValueError("Length of `output_index_names` does not match the number of output columns")
	return dims_equal, n_models, n_obs
