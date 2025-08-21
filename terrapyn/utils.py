import typing as T

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def split_into_chunks(a: T.Iterable = None, n: int = 2, axis: int = 0) -> np.ndarray:
	"""
	Split an array/list/iterable into chunks of length `n`, where the last chunk may have length < n.

	Args:
		a: Iterable array/list
		n: Length of each chunk
		axis: The axis along which to split the data
	Returns:
		Array split into chunks of length `n`

	Example:
		>>> split_into_chunks(np.arange(10), 3)
		[array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8]), array([9])]
	"""
	return np.split(a, range(n, len(a), n), axis=axis)


def split_number_into_parts(a: int = None, b: int = None):
	"""
	Split an integer `a` into a list of `b` integer parts, where
	the values are as equal as possible while remaining whole numbers.
	"""
	return [a // b + 1] * (a % b) + [a // b] * (b - a % b)


def set_dim_values_in_data(
	data: xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series = None,
	values: T.Iterable = None,
	dim: str = None,
) -> xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series:
	"""
	Replaces the values of a dimension/variable/column/index named `dim` in
	xarray or pandas data structures with the values in the iterable `values`.

	Args:
		data: Input data with a column/dimension/index called `dim`.
		values: Values to use to replace the existing values for that index/dim/column.
		dim: Name of dimension/column/index (ignored for `pandas.Series`).

	Returns:
		Data with `dim` values replaced with `values`.
	"""
	if isinstance(data, pd.Series | pd.DataFrame):
		if dim in data.index.names:
			data.index = data.index.set_levels(values, level=dim) if isinstance(data.index, pd.MultiIndex) else values
		else:
			if isinstance(data, pd.Series):
				# Update values of series, not index
				data.update(values)
			else:
				# update column of dataframe
				data[dim] = values
	elif isinstance(data, xr.Dataset | xr.DataArray):
		data = data.assign_coords({dim: values})
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
		raise TypeError(f"Data must be one of type: {data_types_str}")
	return data


def get_dim_values_from_data(
	data: xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series | pd.MultiIndex = None,
	dim: str = None,
) -> np.ndarray:
	"""
	Retrieve the values of a dimension/variable/column/index in xarray or pandas data structures.
	If values are a pd.Index or pd.Series data type, these types are returned, otherwise a np.ndarray is returned.

	Args:
		data: Input data
		dim: Name of dimension/variable/column/index, the values of which will be returned

	Returns:
		Values for the dimension/variable/column/index `dim`
	"""
	# Validate `dim` string
	if not isinstance(dim, str) or len(dim) == 0:
		raise ValueError("`dim` must be a string with length > 0")

	if isinstance(data, pd.Series | pd.DataFrame):
		# Check if `dim` is in the index
		if dim in data.index.names:
			return data.index.get_level_values(dim).to_numpy()
		else:
			if isinstance(data, pd.Series):
				return data.to_numpy()
			elif dim in data.columns:
				return data[dim].to_numpy()
			else:
				raise ValueError(f"`dim` of '{dim} not found in data")
	elif isinstance(data, xr.Dataset | xr.DataArray):
		if dim in data.variables:
			return data.variables[dim].to_numpy()
		else:
			raise ValueError(f"`dim` of '{dim} not found in data")
	elif isinstance(data, pd.MultiIndex):
		# Check if `dim` is the index
		if dim in data.names:
			return data.get_level_values(dim).to_numpy()
		else:
			raise ValueError(f"`dim` of {dim} not found in pd.Multindex")
	else:
		data_types_str = ", ".join(str(i) for i in [xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, pd.Multindex])
		raise TypeError(f"Data is of type {type(data)} but must be one of type: {data_types_str}")


def pandas_to_geopandas(
	df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon", crs=None, **kwargs
) -> gpd.GeoDataFrame:
	"""
	Convert a pandas.DataFrame to a geopandas.GeoDataFrame, adding a new geometry column
	with GeometryArray of shapely Point geometries.

	Args:
		df: Input Pandas dataframe with columns of latitude and longitude.
		lat_col: Name of column with latitudes
		lon_col: Name of column with longitudes
		crs: (Optional) Coordinate Reference System of the geometry objects

	Returns:
		Geopandas GeoDataFrame
	"""
	geometry = gpd.points_from_xy(df[lon_col], df[lat_col], crs=crs)
	gdf = gpd.GeoDataFrame(df, geometry=geometry, **kwargs)
	return gdf


def _pandas_check_multiindex_type(index):
	"""Checks type of index and returns True if pd.Multiindex"""
	return isinstance(index, pd.MultiIndex)


def _dim_in_pandas_index(index, dim):
	"""Check if dimension `dim` is in a Pandas index (of any type)"""
	is_multiindex = _pandas_check_multiindex_type(index)

	if is_multiindex:
		if index.names is not None:
			return dim in index.names
	else:
		if index.name is not None:
			return dim in index.name


def ensure_list(a: T.Any = None) -> list[T.Any]:
	"""
	Ensure data `a` is a list if not None
	"""
	if a is None:
		return None
	elif isinstance(a, list):
		return a
	elif isinstance(a, str | int | float):
		return [a]
	else:
		return list(a)


def _call_resample_method(obj, method, **kwargs):
	"""Apply resample method to grouped object, either pandas"""
	if method == "sum":
		return obj.sum(**kwargs)
	elif method == "mean":
		return obj.mean(**kwargs)
	elif method == "min":
		return obj.min(**kwargs)
	elif method == "max":
		return obj.max(**kwargs)
	# elif method == "cumsum":
	else:
		raise ValueError(f"method=`{method}` not implemented")


def remove_list_elements(input_list: list = None, remove_list: list = None) -> list:
	"""Removes all elements in `remove_list` from `input_list` and returns new list object"""
	if input_list is not None and remove_list is not None:
		# convert all single values to a list
		remove_list = ensure_list(remove_list)

		return [e for e in input_list if e not in set(remove_list)]

	else:
		raise ValueError("both `input_list` and `remove_list` must be provided")


def get_key_for_value_in_dict(dictionary: dict, value):
	"""Get the key for a value in a dictionary"""
	return list(dictionary.keys())[list(dictionary.values()).index(value)]


def get_first_dictionary_value(dictionary: dict):
	"""Get the first value in a dictionary"""
	return next(iter(dictionary.values()))


def get_indexes_of_items_in_list(input_list: list = None, items: list = None) -> list[int]:
	"""Get the (first) index of each item in `items` in the list `input_list`"""
	return [input_list.index(item) for item in items]


def utf8_len(s: str) -> int:
	"""Calculate the size of a string in bytes."""
	return len(s.encode("utf-8"))


def ensure_string(s: str | T.Iterable) -> str:
	"""Accepts a string or an iterable, and returns a string or the first element of the iterable"""
	if isinstance(s, str):
		return s
	elif isinstance(s, T.Iterable):
		return s[0]
