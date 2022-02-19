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


def set_dim_values_in_data(
    data: T.Union[
        xr.Dataset,
        xr.DataArray,
        pd.DataFrame,
        pd.Series,
    ] = None,
    values: T.Iterable = None,
    dim: str = None,
) -> T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series]:
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
    if isinstance(data, (pd.Series, pd.DataFrame)):
        if dim in data.index.names:
            if isinstance(data.index, pd.MultiIndex):
                data.index = data.index.set_levels(values, level=dim)
            else:
                data.index = values
        else:
            if isinstance(data, pd.Series):
                # Update values of series, not index
                data.update(values)
            else:
                # update column of dataframe
                data[dim] = values
    elif isinstance(data, (xr.Dataset, xr.DataArray)):
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


def pandas_to_geopandas(
    df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon", crs=None, *args, **kwargs
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
    gdf = gpd.GeoDataFrame(df, *args, **kwargs)
    gdf["geometry"] = gpd.points_from_xy(gdf[lon_col], gdf[lat_col], crs=crs)
    return gdf
