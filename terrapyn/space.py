import typing as T

import geopandas as gpd
import numpy as np
import odc.geo.xr  # noqa
import pandas as pd
import xarray as xr
from odc.geo import Geometry
from pyproj import Geod
from scipy.spatial import cKDTree

import terrapyn as tp


def get_data_at_coords(
	data: xr.Dataset | xr.DataArray = None,
	lats: T.Iterable = None,
	lons: T.Iterable = None,
	point_names: T.Iterable = None,
	point_names_dim: str = "id",
	method: str = "nearest",
	lat_dim: str = "lat",
	lon_dim: str = "lon",
	time_dim: str = "time",
	ignore_nan: bool = False,
	as_dataframe: bool = True,
	**kwargs,
) -> xr.Dataset | xr.DataArray | pd.DataFrame:
	"""
	Retrieve values for the specified coordinates from an xr.Dataset/Array, for all time steps.

	Args:
		data: Dataset/DataArray of source data.
		lat_dim: Name of latitude dimension in source data.
		lon_dim: Name of longitude dimension in source data.
		time_dim: Name of time dimension in source data.
		method: Method to use to select the values. If ``nearest`` and ``ignore_nan==True`` this uses a
		custom function, otherwise ``method`` is passed to `scipy.interpolate.interp1d`, where addition
		kwargs are passed to scipy via the `**kwargs` argument.
		point_names: Array of unique identifiers for points.
		point_names_dim: Name of the index for the `point_names`, default=`id`.
		lats: Array of latitudes. Not required if `metadata` is provided.
		lons: Array of longitudes. Not required if `metadata` is provided.
		as_dataframe: Whether to return data as a pandas.DataFrame or an xr.Dataset/Array

	Returns:
		Data at the given coordinates, either as a xr.Dataset/DataArray or pandas.DataFrame.

	Example: Retrieve the nearest non-NaN points

		>>> n_time = 5
		>>> data = np.full((n_time, 4, 4), np.nan)
		>>> data[:, 1, 1] = 1
		>>> data[:, 1, 2] = 2
		>>> data[:, 2, 1] = 3
		>>> data[:, 2, 2] = 4
		>>> data[2, 2, 2] = np.nan
		>>> da = xr.DataArray(
		...     data,
		...     coords={
		...	 "lat": ("lat", range(4)),
		...	 "lon": ("lon", range(4, 8)),
		...	 "time": ("time", pd.date_range("01-01-2001", periods=n_time, freq="D")),
		...     },
		...     dims=["time", "lat", "lon"],
		...     name="test",
		... )
		>>> lats = [0, 3]
		>>> lons = [4, 6]
		>>> point_names = ["a", "b"]
		>>> get_data_at_coords(  # doctest: +NORMALIZE_WHITESPACE
		...     da, lats=lats, lons=lons, point_names=point_names, method="nearest", ignore_nan=True
		... )
			       test
		time       id
		2001-01-01 a    1.0
			   b    4.0
		2001-01-02 a    1.0
			   b    4.0
		2001-01-03 a    1.0
			   b    3.0
		2001-01-04 a    1.0
			   b    4.0
		2001-01-05 a    1.0
			   b    4.0
	"""  # noqa: E101
	if lats is None or lons is None:
		raise ValueError("Must provide `lats` and `lons`")

	# Retrieve nearest non-NaN values, for each coordinate that is not lat or lon (so time etc.)
	if method == "nearest" and ignore_nan:
		# Convert dataarray to dataframe
		points = data.to_dataframe().dropna()

		non_lat_lon_index_names = list(points.index.names.difference([lat_dim, lon_dim]))

		points = points.groupby(non_lat_lon_index_names).apply(
			get_nearest_point,
			lats=lats,
			lons=lons,
			lat_dim=lat_dim,
			lon_dim=lon_dim,
			return_distance=False,
			reset_index=True,
			index_name=point_names_dim,
		)

	# Retrieve nearest values (including NaN) for each coordinate that is not lat or lon (so time etc.)
	else:
		lat_idx = xr.DataArray(lats, dims=point_names_dim)
		lon_idx = xr.DataArray(lons, dims=point_names_dim)
		points = data.interp({lat_dim: lat_idx, lon_dim: lon_idx}, method=method, kwargs=kwargs)

		# Convert xr.DataArray to pd.DataFrame
		points = points.to_dataframe()

	# Drop lat and lon columns
	if lat_dim in points.columns:
		points = points.drop(columns=lat_dim)
	if lon_dim in points.columns:
		points = points.drop(columns=lon_dim)

	# Rename labels for point coordinates to `point_names`
	if point_names is not None:
		if isinstance(points.index, pd.MultiIndex):
			points.index = points.index.set_levels(point_names, level=point_names_dim)
		else:
			points.index = pd.Index(point_names, name=point_names_dim)

	if as_dataframe:
		# Return dataframe with `time_dim` and `point_names_dim` as the index (if both exist),
		# with all other coordinates reset to columns.
		index_coords_to_reset = list(points.index.names.difference([time_dim, point_names_dim]))
		if len(index_coords_to_reset) > 0:
			points = points.reset_index(index_coords_to_reset)

		if time_dim in points.index.names:
			# Make sure time is the first index
			dim_order = [time_dim, point_names_dim]
			points.index = points.index.reorder_levels(dim_order)
		else:
			dim_order = [point_names_dim]
		points = points.sort_index(level=dim_order)
	else:
		points = points.to_xarray()
	return points


def _geodesic_distances_between_point_and_endpoints(point: tuple[float, float], endpoints):
	"""
	Return the geodesic distances in kilometres (km) between `point` and the points in `endpoints`.
	`point` and `endpoints` have column order of (lat, lon)
	"""
	endpoints = np.asarray(endpoints)

	# Check array shapes are valid
	if len(point) != 2:
		raise ValueError("`point` must be a tuple of (lat, lon)")
	if len(endpoints.shape) != 2 & endpoints.shape[1] != 2:
		raise ValueError("`endpoints` must be an n x 2 array")

	point = np.full(endpoints.shape, point)
	geod = Geod(ellps="WGS84")
	distance_km = geod.inv(point[:, 1], point[:, 0], endpoints[:, 1], endpoints[:, 0])[2] / 1000
	return distance_km


def _index_and_distance_of_nearest_point(point, endpoints):
	"""
	Return the index and geodesic distance in km of the nearest point in `endpoints`, to the coordinate `point`.
	`point` and `endpoints` have column order of (lat, lon)
	"""
	distances_km = _geodesic_distances_between_point_and_endpoints(point, endpoints)
	index_of_nearest = np.argmin(distances_km)
	return index_of_nearest, distances_km[index_of_nearest]


def get_nearest_point(
	df: pd.DataFrame,
	lats: float | T.Iterable,
	lons: float | T.Iterable,
	lat_dim: str = "lat",
	lon_dim: str = "lon",
	method: str = "geodesic",
	return_distance: bool = True,
	reset_index: bool = True,
	index_name: str = "point",
) -> pd.DataFrame:
	"""
	Return nearest points from a pandas.DataFrame

	Args:
		df: Dataframe with values at coordinates
		lats: Array of latitudes at which to select nearest points.
		lons: Array of longitudes at which to select nearest points.
		lat_dim: Name of latitude index/column in data.
		lon_dim: Name of longitude index/colimn in data.
		method: Method to use to retrieve points, one of 'kdtree' or 'geodesic'. If 'geodesic', the
		distance can optionally be returned with `return_distance==True`
		return_distance: If `True`, add a column of distance (in km) from the source point to the target.
		Only applies when `method=='geodesic'`.
		reset_index: Whether to reset the dataframe index values or not.
		index_name: The name for the output index.

	Returns:
		A dataframe with the nearest points to the desired `lats` and `lons` coordinates
	"""
	if isinstance(lats, float | int):
		lats = [lats]
	if isinstance(lons, float | int):
		lons = [lons]

	# order of (lat, lon)
	points = np.column_stack([lats, lons])
	endpoints = np.column_stack(get_lat_lon_from_data(df, lat_name=lat_dim, lon_name=lon_dim, unique=False))

	if method == "kdtree":
		tree = cKDTree(endpoints)
		_, indices = tree.query(points, k=1)

	elif method == "geodesic":
		# For each point, find the nearest endpoint and add the index to a list
		indices = []
		distances = []
		for p in points:
			index_of_nearest, distance_of_nearest = _index_and_distance_of_nearest_point(p, endpoints)
			indices.append(index_of_nearest)
			distances.append(distance_of_nearest)

	df = df.iloc[indices].copy()

	if return_distance and method == "geodesic":
		df["distance_km"] = distances

	if reset_index:
		df = df.reset_index(drop=True)
	if index_name is not None:
		df.index = df.index.rename(index_name)

	return df


def get_lat_lon_from_data(
	data: xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series | gpd.GeoDataFrame = None,
	lon_name: str = "lon",
	lat_name: str = "lat",
	unique: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
	"""
	Extract latitude and longitude coordinates from data

	Args:
		data: Input data
		lon_name: Name of longitude dimension/column
		lat_name: Name of latitude dimension/column
		unique: Return only unique pairs of lat and lon - useful for pandas.DataFrame where coordinates are repeated

	Returns:
		Tuple of arrays of lat and lon
	"""
	if isinstance(data, xr.Dataset | xr.DataArray):
		lats = data[lat_name].to_numpy()
		lons = data[lon_name].to_numpy()
	elif isinstance(data, pd.DataFrame | pd.Series | gpd.GeoDataFrame):
		if lon_name in data.index.names and lat_name in data.index.names:
			if unique:
				lats = data.index.unique(lat_name).to_numpy()
				lons = data.index.unique(lon_name).to_numpy()
			else:
				lats = data.index.get_level_values(lat_name).to_numpy()
				lons = data.index.get_level_values(lon_name).to_numpy()
		elif lon_name in data.columns and lat_name in data.columns:
			if unique:
				lats = data[lat_name].unique()
				lons = data[lon_name].unique()
			else:
				lats = data[lat_name].to_numpy()
				lons = data[lon_name].to_numpy()
		else:
			raise ValueError(f"{lat_name} and {lon_name} were not found in the data")
	return lats, lons


def crop(
	data: xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series | gpd.GeoDataFrame = None,
	geopolygon: Geometry = None,
	apply_mask: bool = False,
	all_touched: bool = True,
	lon_name: str = "lon",
	lat_name: str = "lat",
	geometry_name: str = "geometry",
	crs="EPSG:4326",
) -> xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series | gpd.GeoDataFrame:
	"""
	Return all data within the given geobox. Accepts xarray dataset/array or pandas/geopandas dataframe.
	Latitude can be ordered positive to negative, or negative to positive.

	Args:
		data: Input data
		geopolygon: A odc.geo.geom.Geometry polygon used to crop the data. If `None`, data is returned unaltered.
		apply_mask: Whether to mask out pixels outside of the rasterized extent of poly by setting them to NaN.
		all_touched: If True and apply_mask=True, the rasterize step will burn in all pixels touched by geopolygon.
		If False, only pixels whose centers are within the polygon or that are selected by Bresenham’s line algorithm
		will be burned in.
		lon_name: Name of longitude dimension/column.
		lat_name: Name of latitude dimension/column.
		geometry_name: Name of column containing coordinates of shapely point objects in geopandas.GeoDataFrame.
		crs: Coordinate reference system of the data (if not given).

	Returns:
		Data cropped / masked to the given polygon.
	"""
	if geopolygon is None:
		return data

	if isinstance(data, xr.Dataset | xr.DataArray):
		_crs = getattr(data, "spatial_ref", None)
		if _crs is None:
			_crs = getattr(data, "crs", crs)
			data = data.odc.assign_crs(crs=_crs)
		return data.odc.crop(poly=geopolygon, apply_mask=apply_mask, all_touched=all_touched)

	elif isinstance(data, pd.DataFrame | pd.Series | gpd.GeoDataFrame):
		# If data is geopandas.geodataframe, first try to clip by geometry column
		if isinstance(data, gpd.GeoDataFrame) and geometry_name in data.columns:
			# Ensure CRS of data and geopolygon are the same
			data = data.set_crs(crs) if data.crs is None else data.to_crs(geopolygon.crs)
			return data.clip(geopolygon.geom)

		# If data is pandas dataframe/series or geopandas dataframe without geometry
		lats, lons = get_lat_lon_from_data(data, lon_name=lon_name, lat_name=lat_name, unique=False)

		# Generate points for each coordinate and use spatial indexing to find points within geopolygon
		coordinates = gpd.points_from_xy(lons, lats)
		return data.loc[coordinates.within(geopolygon.geom)]
	else:
		raise ValueError("Coordinates not found. Either `lon_name` and `lat_name` or `geometry_name` must be provided")


def data_bounds(
	data: xr.Dataset | xr.DataArray | pd.DataFrame | pd.Series | gpd.GeoDataFrame = None,
	lon_name: str = "lon",
	lat_name: str = "lat",
	use_pixel_bounds: bool = False,
	decimals: int = 9,
) -> tuple:
	"""
	Retrieve bounds from input data.

	Args:
		data: Input data
		lon_name: Name of longitude dimension/column
		lat_name: Name of latitude dimension/column
		use_pixel_bounds: If `True` the boundary will be the bounds of the outermost coordinates in the
		data, otherwise the bounds will be the pixel centers.
		decimals: Number of decimal places to return (using `np.around`). A value of `9` is
		around 1 cm on the Earth's surface if the coordinates are in degrees.

	Returns:
		Returns Tuple of SW and NE corners in lat/lon order, ((lat_s, lon_w), (lat_n, lon_e))
	"""
	lats, lons = get_lat_lon_from_data(data, lon_name, lat_name, unique=True)

	if use_pixel_bounds:
		lats = get_coordinate_bounds_from_centers(lats)
		lons = get_coordinate_bounds_from_centers(lons)

	min_lat = np.around(lats.min(), decimals=decimals)
	max_lat = np.around(lats.max(), decimals=decimals)
	min_lon = np.around(lons.min(), decimals=decimals)
	max_lon = np.around(lons.max(), decimals=decimals)

	return ((min_lat, min_lon), (max_lat, max_lon))


def generate_grid(
	left: float = -180,
	bottom: float = -90,
	right: float = 180,
	top: float = 90,
	resolution: float = 1.0,
	return_type: str = "dataarray",
	fill_value: int | float = None,
) -> tuple[np.ndarray, np.ndarray] | xr.DataArray | xr.Dataset:
	"""
	Makes an xr.DataArray/xr.Dataset grid with optional variable and fill value.

	Args:
		resolution: Resolution.
		return_type: Type of object to return; 'numpy' for arrays of latitude, longitude, 'dataarray'
		for Xarray DataArray.
		fill_value: Value used to fill the grid. Defaults to `np.nan` if `fill_value` is `None`.

	Returns:
		Arrays or DataArray with the given coordinates, with optional fill value.
	"""
	lats = np.arange(bottom, top + resolution, resolution)
	lons = np.arange(left, right + resolution, resolution)

	if return_type == "numpy":
		return lats, lons

	data = np.full(shape=(lats.shape[0], lons.shape[0]), fill_value=fill_value) if fill_value else np.nan

	return xr.DataArray(data=data, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"])


def points_within_radius(
	df: pd.DataFrame = None,
	point: tuple[float, float] = None,
	radius_km: float = 100,
	lat_col: str = "lat",
	lon_col: str = "lon",
	return_distance: bool = True,
) -> pd.DataFrame:
	"""
	Return a subset of dataframe for coordinates that lie within the geodesic distance
	between coordinates (for accurate distances on Earth's surface).

	Args:
		df: Dataframe with unique identifier and coordinates
		point: Coordinate of source point in order (lat, lon)
		lat_col: Name of latitude column in dataframe.
		lon_col: Name of longitude column in dataframe.
		radius_km: Radius of circle in kilometres (km).
		return_distance: If `True`, add a column of distance (in km) from the source point to the target.

	Returns:
		Subset of dataframe with points inside the given radius.
	"""
	endpoints = np.column_stack(get_lat_lon_from_data(df, lat_name=lat_col, lon_name=lon_col, unique=False))
	distances_km = _geodesic_distances_between_point_and_endpoints(point, endpoints)
	indices_within_radius = distances_km < radius_km
	df_within_radius = df.iloc[indices_within_radius].copy()
	if return_distance:
		df_within_radius["distance_km"] = distances_km[indices_within_radius]
	return df_within_radius


def add_coordinate_bounds(
	data: xr.Dataset | xr.DataArray, lat_name: str = "lat", lon_name: str = "lon"
) -> xr.Dataset | xr.DataArray:
	"""
	Add coordinate bounds 'lon_b' and 'lat_b' to an input dataset/array

	Args:
		data: Dataset/Array containing coordinates `lat_name` and `lon_name`
		lat_name: Name of the lat coordinate
		lon_name: Name of the lon coordinate

	Returns:
		The original data with new coordinates of `lat_b` and `lon_b` that give
		the boundaries of the input coordinates.
	"""
	lons = data[lon_name].to_numpy()
	lats = data[lat_name].to_numpy()
	lat_bounds = get_coordinate_bounds_from_centers(lats)
	lon_bounds = get_coordinate_bounds_from_centers(lons)
	data["lon_b"] = lon_bounds
	data["lat_b"] = lat_bounds
	return data


def get_coordinate_bounds_from_centers(coords: np.ndarray = None) -> np.ndarray:
	"""
	Return the boundaries (edges) of each cell/pixel from an array of center values, where we
	assume the difference between the first 2 values is the step size for all coordinates.
	This is useful to generate `lon_b` and `lat_b` coordinates for regridding purposes.

	Args:
		coords: Array of coordinates, the centers of the pixels/cells.

	Returns:
		Array of the coordinates boundaries, where there are `len(coords) + 1` elements.
	"""
	step = coords[1] - coords[0]
	pad = step / 2.0
	bounds = np.linspace(coords[0] - pad, coords[-1] + pad, len(coords) + 1)
	return bounds


def get_coordinate_centers_from_bounds(coord_bounds: np.ndarray = None) -> np.ndarray:
	"""
	Return the center of the pixels/cells from an array of pixel/cell bounds.

	Args:
		coord_bounds: Array of the boundaries of pixels/cells

	Returns:
		Array of the coordinate centers, where there are `len(coord_bounds) - 1` elements.
	"""
	step = coord_bounds[1] - coord_bounds[0]
	pad = step / 2.0
	centers = coord_bounds[:-1] + pad
	return centers


def group_points_by_grid(
	df: pd.DataFrame = None,
	lat_col: str = "lat",
	lon_col: str = "lon",
	id_col: str = "id",
	cellsize: float = 5,
	return_cell_bounds: bool = True,
	lat_bounds: T.Iterable = None,
	lon_bounds: T.Iterable = None,
) -> pd.DataFrame:
	"""
	Group points together based on location, using a grid. Each square in the grid has length
	'cellsize' in degrees, and the grid's bounds are defined by the outermost point coordinates.

	Args:
		df: Dataframe with point names and coordinates.
		lat_col: Name of latitude column in dataframe.
		lon_col: Name of longitude column in dataframe.
		id_col: Name of id (station name etc.) column in dataframe.
		cellsize: Size of each grid square in degrees, in which to group the points.
		return_cell_bounds: If `True` return additionally return arrays of the grid cell bounds
		lat_bounds: Array of grid boundaries for latitudes. If given, takes precedence over `cellsize`.
		lon_bounds: Array of grid boundaries for longitudes. If given, takes precedence over `cellsize`.

	Returns:
		pd.Series of groups of points (with integer labels) and a list of points in that group
	"""
	lats, lons = get_lat_lon_from_data(df, lat_name=lat_col, lon_name=lon_col, unique=False)

	if lat_bounds is None and lon_bounds is None:
		min_lat, max_lat = lats.min(), lats.max()
		min_lon, max_lon = lons.min(), lons.max()

		# grid cells should extend cellsize * 0.5 beyond the outermost stations so stations are centered in cells
		pad = cellsize / 2.0

		lat_bins = np.arange(min_lat - pad, max_lat + pad + cellsize, cellsize)
		lon_bins = np.arange(min_lon - pad, max_lon + pad + cellsize, cellsize)
	else:
		lat_bins = lat_bounds
		lon_bins = lon_bounds

	lat_labels = np.digitize(lats, lat_bins)
	lon_labels = np.digitize(lons, lon_bins)

	groups = df.groupby([lat_labels, lon_labels])[id_col].apply(list)
	groups = groups.reset_index(drop=True).rename("group")

	if return_cell_bounds:
		return groups, lat_bins, lon_bins
	else:
		return groups


def inverse_distance_weighting(
	data: pd.DataFrame | pd.Series | gpd.GeoDataFrame = None,
	lon_name: str = "lon",
	lat_name: str = "lat",
	value_name: str = "var",
	lons: pd.Series | np.ndarray | list = None,
	lats: pd.Series | np.ndarray | list = None,
	values: pd.Series | np.ndarray | list = None,
	lons_out: np.ndarray = None,
	lats_out: np.ndarray = None,
	k: int = 12,
	p: int = 2,
	eps: float = 1e-6,
	distance_upper_bound: float = 1e6,
	regularize_by: float = 1e-9,
	leafsize: int = 16,
	return_type: str = "dataarray",
) -> xr.DataArray:
	r"""
	Calculate the inverse-distance-weighted mean of (ir)regularly spaced 2-D data and return the mean on a regularly
	spaced grid.

	For a given 'p' (p-norm or power), for a given coordinate, the
	mean value :math:`z_{p}` is calculated as:

	.. math::
		z_{p} = \\frac{\\\
		\\displaystyle\\sum_{i=1}^{k}\\left(\\frac{z_{i}}{d^{p}_{i}}\\right)}{\\\
		\\displaystyle\\sum_{i=1}^{k}\\left(\\frac{1}{d^{p}_{i}}\\right)}

	where :math:`d` is the distance to the nearest point :math:`i` that has
	a value of :math:`z`

	Args:
		data: Input data containing indexes/columns of `lon_name`, `lat_name` and `value_name`
		lon_name: Name of longitude column/index
		lat_name: Name of latitude column/index
		value_name: Name of values column
		lons: 1-D array of longitude coordinates (if `data` is not given).
		lats: 1-D array of latitude coordinates (if `data` is not given).
		values: Values at each lon, lat coordinate, 1-D array  (if `data` is not given).
		lon_out: 1-D array of output longitude coordinates.
		lat_out: 1-D array of output latitude coordinates.
		k: The number of nearest neighbors to include. Must be >1.
		p: Which Minkowski p-norm to use, where 1 is the sum of absolute values ("Manhattan" distance),
		2 is the Euclidean distance, infinity is the maximum-coordinate-difference distance. Must be positive.
		eps: Factor to allow approximate nearest neighbors, such that the k-th returned value is guaranteed to
		be no further than (1 + eps) times the distance to the real k-th nearest neighbor.
		distance_upper_bound: Return only neighbors within this distance. Default is 1000 which includs all
		points if longitude and latitude are in degrees.
		regularize_by: Regularize distances to prevent division by zero for sample points with the same
		location as query points.
		leafsize: The number of points at which the algorithm switches over to brute-force. Default==16. Must be >=1.
		return_type: Format of the output. If 'dataarray' return an xr.DataArray, if 'dataframe' return a pd.DataFrame,
		if 'numpy' return a 2-D numpy array.

	Returns:
		Array of values on a regular grid that are the inverse-distance-weighted mean of the given input values.
	"""
	if lons_out is None or lats_out is None:
		raise ValueError("Both `lons_out` and `lats_out` must be given")

	if not isinstance(k, int):
		raise TypeError(f"k is of type {type(k)} but must be of type {int}")
	if k < 2:
		raise ValueError("k must be > 1")

	if not isinstance(p, float | int):
		raise TypeError(f"p is of type {type(p)} but must be of type {int} or {float}")
	if p < 0.0:
		raise ValueError("p must be positive (>0)")

	if not isinstance(distance_upper_bound, float | int):
		raise TypeError(f"distance_upper_bound is of type {type(p)} but must be of type {int} or {float}")
	if distance_upper_bound < 0.0:
		raise ValueError("distance_upper_bound must be positive (>0)")

	if not isinstance(leafsize, int):
		raise TypeError(f"leafsize is of type {type(leafsize)} but must be of type {int}")
	if leafsize < 1:
		raise ValueError("leafsize must be >= 1")

	if data is None:
		lats = np.array(lats)
		lons = np.array(lons)
		values = np.array(values)
	else:
		lats, lons = tp.space.get_lat_lon_from_data(data, lon_name=lon_name, lat_name=lat_name, unique=False)
		values = data[value_name].to_numpy()

	if lons.ndim > 1:
		raise TypeError(f"`lons` array is not 1-D (ndim={lons.ndim})")
	if lats.ndim > 1:
		raise TypeError(f"`lats` array is not 1-D (ndim={lats.ndim})")
	if values.ndim > 1:
		raise TypeError(f"`values` array is not 1-D (ndim={values.ndim})")

	# Build input array and calculate tree of points
	X = np.stack([lons, lats], axis=1)
	tree = cKDTree(X, leafsize=leafsize)

	# Generate output grid and reshape array to have 2 columns of x and y
	X_out = np.meshgrid(lons_out, lats_out)
	grid_shape = X_out[0].shape
	X_out = np.reshape(X_out, (2, -1)).T

	# Calculate distances and weights
	distances, idx = tree.query(X_out, k=k, eps=eps, p=p, distance_upper_bound=distance_upper_bound)
	distances = distances + regularize_by
	weights = values[idx.ravel()].reshape(idx.shape)

	# Calculate mean weights
	mean_weight = np.sum(weights / distances, axis=1) / np.sum(1.0 / distances, axis=1)

	if return_type == "numpy":
		return mean_weight.reshape(grid_shape)

	elif return_type == "dataarray":
		return xr.DataArray(
			data=mean_weight.reshape(grid_shape), dims=["lat", "lon"], coords={"lon": lons_out, "lat": lats_out}
		)

	elif return_type == "dataframe":
		return pd.DataFrame(
			data=mean_weight,
			index=pd.MultiIndex.from_arrays([X_out[:, 1], X_out[:, 0]], names=["lat", "lon"]),
			columns=[value_name],
		)
	else:
		raise ValueError("return_type must be one of 'numpy', 'dataarray', 'dataframe'")
