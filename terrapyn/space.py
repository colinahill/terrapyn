import typing as T

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
import xarray as xr
from pyproj import Geod
from scipy.spatial import cKDTree


class BBox:
    """
    Object to define and store a bounding box. Coordinates can be user-defined by passing
    coordinates, or by passing a geometry. Bounding coordinates are stored as a shapely polygon,
    accessible with the `geom` attribute.

    By default, bounding box is set to the entire globe

    Attributes:
        min_lon (float): The minimum longitude
        max_lon (float): The maximum longitude
        min_lat (float): The minimum latitude
        max_lat (float): The maximum latitude
        geom (shapely.geometry): The shapely.geometry object that defines the bounding box.
        bounds: Returns the outer bounds of the geometry (a rectangle).
        WNES (tuple): Tuple of bounding coordinates in order of West, North, East, South.
        NWSE (tuple): Tuple of bounding coordinates in order of North, West, South, East.
    """

    def __init__(self, geometry=None, min_lon=-180, max_lon=180, min_lat=-90, max_lat=90):
        if geometry is None:
            self.geom = shapely.geometry.box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)
        else:
            if not isinstance(geometry, shapely.geometry.base.BaseGeometry):
                raise TypeError("`geometry` must be a shapely geometry type")
            self.geom = geometry

        self.min_lon, self.min_lat, self.max_lon, self.max_lat = self.geom.bounds

    def __repr__(self):
        return (
            f"BBox(min_lon={self.min_lon}, max_lon={self.max_lon}, " f"min_lat={self.min_lat}, max_lat={self.max_lat})"
        )

    @property
    def bounds(self):
        return self.geom.bounds

    @property
    def WNES(self):
        return (self.min_lon, self.max_lat, self.max_lon, self.min_lat)

    @property
    def NWSE(self):
        return (self.max_lat, self.min_lon, self.min_lat, self.max_lon)

    @property
    def centroid(self):
        return self.geom.centroid

    @property
    def centroid_coords(self) -> T.Tuple[float, float]:
        """Coordinates of the centroid of bounding region, a tuple of (latitude, longitude)."""
        lon, lat = list(self.geom.centroid.coords)[0]
        return (lat, lon)

    def buffer(self, distance=0, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0, single_sided=False):
        """
        Buffer/Expand BBox geometry following Shapely buffer method and return new BBox. Does not modify
        original BBox geometry. https://shapely.readthedocs.io/en/latest/manual.html#object.buffer
        """
        return BBox(
            geometry=self.geom.buffer(
                distance=distance,
                resolution=resolution,
                cap_style=cap_style,
                join_style=join_style,
                mitre_limit=mitre_limit,
                single_sided=single_sided,
            )
        )


def get_data_at_coords(
    data: T.Union[xr.Dataset, xr.DataArray] = None,
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
) -> T.Union[xr.Dataset, xr.DataArray, pd.DataFrame]:
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
        >>> da = xr.DataArray(data,
        ...                   coords={'lat':('lat', range(4)), 'lon': ('lon', range(4, 8)),
        ...                   'time': ('time', pd.date_range("01-01-2001", periods=n_time, freq='D'))},
        ...                   dims=['time', 'lat', 'lon'], name='test')
        >>> lats = [0, 3]
        >>> lons = [4, 6]
        >>> point_names = ['a', 'b']
        >>> get_data_at_coords(  # doctest: +NORMALIZE_WHITESPACE
        ...                    da, lats=lats, lons=lons, point_names=point_names,
        ...                    method='nearest', ignore_nan=True)
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
    """
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
        points.drop(columns=lat_dim, inplace=True)
    if lon_dim in points.columns:
        points.drop(columns=lon_dim, inplace=True)

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
            dim_order = [time_dim, point_names_dim]
            points.index = points.index.reorder_levels(dim_order)
        else:
            dim_order = [point_names_dim]
        points = points.sort_index(level=dim_order)
    else:
        points = points.to_xarray()
    return points


def _geodesic_distances_between_point_and_endpoints(point: T.Tuple[float, float], endpoints):
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
    lats: T.Union[float, T.Iterable],
    lons: T.Union[float, T.Iterable],
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
    if isinstance(lats, (float, int)):
        lats = [lats]
    if isinstance(lons, (float, int)):
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
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, gpd.GeoDataFrame] = None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    unique: bool = True,
) -> T.Tuple[np.ndarray, np.ndarray]:
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
    if isinstance(data, (xr.Dataset, xr.DataArray)):
        lats = data[lat_name].values
        lons = data[lon_name].values
    elif isinstance(data, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
        if lon_name in data.index.names and lat_name in data.index.names:
            if unique:
                lats = data.index.unique(lat_name).values
                lons = data.index.unique(lon_name).values
            else:
                lats = data.index.get_level_values(lat_name).values
                lons = data.index.get_level_values(lon_name).values
        elif lon_name in data.columns and lat_name in data.columns:
            if unique:
                lats = data[lat_name].unique()
                lons = data[lon_name].unique()
            else:
                lats = data[lat_name].values
                lons = data[lon_name].values
        else:
            raise ValueError("{} and {} were not found in the data".format(lat_name, lon_name))
    return lats, lons


def crop_to_bbox(
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, gpd.GeoDataFrame] = None,
    bbox: BBox = None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    geometry_name: str = "geometry",
) -> T.Union[xr.Dataset, xr.DataArray]:
    """
    Return all data within the given bounding box. Accepts xarray dataset/array or pandas/geopandas dataframe.
    Latitude can be ordered positive to negative, or negative to positive.

    Args:
        data: Input data
        BBox: Bounding box for data. If `None`, data is returned unaltered
        lon_name: Name of longitude dimension/column
        lat_name: Name of latitude dimension/column
        geometry_name: Name of column containing coordinates of shapely point objects in geopandas.GeoDataFrame

    Returns:
        Subset of data that is within the given bounding box
    """
    if bbox is None:
        return data
    else:
        # Add additional padding to bbox to correct for rounding errors, where units are assumed
        # to be in degrees, where distance is ~1cm on Earth's surface
        bbox = bbox.buffer(distance=0.0000001)

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        # Select longitude range
        data = data.sel({lon_name: slice(bbox.min_lon, bbox.max_lon)})

        if data[lat_name][0] < data[lat_name][-1]:
            # Latitude Ordered Negative to Positive
            return data.sel({lat_name: slice(bbox.min_lat, bbox.max_lat)})
        else:
            # Latitude Ordered Positive to Negative
            return data.sel({lat_name: slice(bbox.max_lat, bbox.min_lat)})

    elif isinstance(data, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
        # If data is geopandas.geodataframe, first try to select using shapely geometry
        if isinstance(data, gpd.GeoDataFrame):
            if geometry_name in data.columns:
                indexes_within_bbox = data[geometry_name].within(bbox.geom)
                return data.loc[indexes_within_bbox]

        # If data is pandas dataframe/series or geopandas dataframe without shapely geometry
        lats, lons = get_lat_lon_from_data(data, lon_name=lon_name, lat_name=lat_name, unique=False)

        # Generate shapely points for each coordinate and find points within bbox
        coordinates = gpd.points_from_xy(lons, lats)
        indexes_within_bbox = coordinates.within(bbox.geom)
        return data.loc[indexes_within_bbox]
    else:
        raise ValueError("Coordinates not found. Either `lon_name` and `lat_name` or `geometry_name` must be provided")


def match_data_bbox(
    reference_data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, gpd.GeoDataFrame] = None,
    input_data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, gpd.GeoDataFrame] = None,
    ref_lat_name: str = "lat",
    ref_lon_name: str = "lon",
    input_lat_name: str = "lat",
    input_lon_name: str = "lon",
    geometry_name: str = "geometry",
) -> T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, gpd.GeoDataFrame]:
    """
    Crop the `input_data` to match the bounding box of the `output_data`. The data type of `input_data` is not modified.

    Args:
        reference_data: Reference data with coordinates that will be used to crop `input_data`.
        input_data: Input data that will be cropped.
        ref_lat_name: Name of latitude coordinate in `reference_data`.
        ref_lon_name: Name of longitude coordinate in `reference_data`.
        input_lat_name: Name of latitude coordinate in `input_data`.
        input_lon_name: Name of longitude coordinate in `input_data`.
        geometry_name: Name of column containing coordinates of shapely point objects if the data
        type is a geopandas.GeoDataFrame.

    Returns:
        Cropped input data.
    """
    output_bbox = bbox_from_data(reference_data, lat_name=ref_lat_name, lon_name=ref_lon_name)

    # Add additional padding to bbox to correct for rounding errors, ~1cm on Earth's surface
    output_bbox = output_bbox.buffer(distance=0.0000001)

    return crop_to_bbox(input_data, output_bbox, lon_name=input_lon_name, lat_name=input_lat_name)


def bbox_from_data(
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, gpd.GeoDataFrame] = None,
    lon_name: str = "lon",
    lat_name: str = "lat",
    use_pixel_bounds: bool = False,
    decimals: int = 9,
) -> BBox:
    """
    Generate BBox from input data.

    Args:
        data: Input data
        lon_name: Name of longitude dimension/column
        lat_name: Name of latitude dimension/column
        use_pixel_bounds: If `True` the boundary will be the bounds of the outermost coordinates in the
        data, otherwise the bounds will be the pixel centers.
        decimals: Number of decimal places to return (using `np.around`). A value of `9` is
        around 1 cm on the Earth's surface if the coordinates are in degrees.

    Returns:
        BBox Subset of data that is within the given bounding box
    """
    lats, lons = get_lat_lon_from_data(data, lon_name, lat_name, unique=True)

    if use_pixel_bounds:
        lats = get_coordinate_bounds_from_centers(lats)
        lons = get_coordinate_bounds_from_centers(lons)

    min_lat = np.around(lats.min(), decimals=decimals)
    max_lat = np.around(lats.max(), decimals=decimals)
    min_lon = np.around(lons.min(), decimals=decimals)
    max_lon = np.around(lons.max(), decimals=decimals)

    return BBox(max_lat=max_lat, min_lon=min_lon, min_lat=min_lat, max_lon=max_lon)


def generate_grid(
    bbox: BBox = None,
    resolution: float = 1.0,
    min_lat: float = -90,
    max_lat: float = 90,
    min_lon: float = -180,
    max_lon: float = 180,
    variable_name: str = "var",
    fill_value: T.Union[int, float] = None,
    return_dataset: bool = False,
) -> T.Union[xr.DataArray, xr.Dataset]:
    """
    Makes an xr.DataArray/xr.Dataset grid with optional variable and fill value.

    Args:
        resolution: Resolution (in degrees).
        bbox: Optional Bounding box for data. Takes precedence over other coordinates.
        min_lat: Lower latitude.
        max_lat: Upper latitude.
        min_lon: Lower longitude.
        max_lon: Upper longitude.
        variable_name: Optional variable name.
        fill_value: Value used to fill the grid. Defaults to `np.nan` if `fill_value` is `None`.
        return_dataset: If True return an xr.Dataset, otherwise return an xr.DataArray.

    Returns:
        DataArray/set with the given coordinates, with optional fill value and variable name.
    """
    if bbox:
        max_lat, min_lon, min_lat, max_lon = bbox.NWSE
    lats = np.arange(min_lat, max_lat + resolution, resolution)
    lons = np.arange(min_lon, max_lon + resolution, resolution)

    if fill_value:
        # Generate data
        data = np.full(shape=(lats.shape[0], lons.shape[0]), fill_value=fill_value)
    else:
        # Return an empty xr.DataArray
        data = np.nan
    da = xr.DataArray(data=data, coords={"lat": lats, "lon": lons}, dims=["lat", "lon"], name=variable_name)

    if return_dataset:
        return da.to_dataset(name=variable_name)
    else:
        return da


def points_within_radius(
    df: pd.DataFrame = None,
    point: T.Tuple[float, float] = None,
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
    data: T.Union[xr.Dataset, xr.DataArray], lat_name: str = "lat", lon_name: str = "lon"
) -> T.Union[xr.Dataset, xr.DataArray]:
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
    lons = data[lon_name].values
    lats = data[lat_name].values
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
    lats, lons = get_lat_lon_from_data(df, lat_name=lat_col, lon_name=lon_col, unique=True)

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
