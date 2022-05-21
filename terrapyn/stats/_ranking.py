import typing as T

import bottleneck as bn
import numpy as np
import pandas as pd
import xarray as xr


def calculate_quantiles(
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series] = None,
    q: T.Iterable = [0.25, 0.5, 0.75],
    dim: T.Union[str, int] = None,
    numeric_only: bool = False,
    interpolation: str = "linear",
    keep_attrs: bool = False,
    skipna: bool = True,
    add_rank_coord: bool = False,
    starting_rank: int = 1,
) -> T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series]:
    """
    Calculate quantiles for data over a given dimension. Essentially a wrapper around `xarray` and `pandas`
    functions, with the option for adding a `rank` coordinate:
    http://xarray.pydata.org/en/stable/generated/xarray.DataArray.quantile.html
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.quantile.html

    Args:
        data: Input data
        q: Levels at which to calculate the quantiles - must be between 0 and 1 inclusive.
        dim: For `xarray`, the name of dimension over which to calculate quantiles.
        For `pandas`, the axis number (one of `0`, `1`, `'index'`, `'columns'`), with default==0.
        numeric_only: Default==True. If False, the quantile of datetime and timedelta data will be computed as well.
        interpolation: Type of interpolation to use when quantile lies between two data points,
        one of 'linear', 'lower', 'higher', 'midpoint', 'nearest'.
        keep_attrs: Only for `xarray` objects. If True, the dataset’s attributes (attrs) will be copied from the
        original object to the new one. If False (default), the new object will be returned without attributes.
        skipna: Whether to skip missing values when aggregating. Default=True
        add_rank_coord: If True, add a coordinate (`xarray`) or column (`pandas`) with the rank
        value of the quantile (`range(len(q))`)
        starting_rank: Starting value of the rank coordinate/column

    Returns:
        Data with the values of the specified quantiles
    """
    allowed_types = (xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series)
    if not isinstance(data, allowed_types):
        raise TypeError("`data` must be of type: {:s}".format(", ".join(str(obj) for obj in allowed_types)))

    if add_rank_coord:
        rank = np.arange(starting_rank, starting_rank + len(q), dtype=int)

    if isinstance(data, (xr.Dataset, xr.DataArray)):
        if isinstance(data, xr.DataArray):
            quantiles = data.quantile(q=q, dim=dim, interpolation=interpolation, keep_attrs=keep_attrs, skipna=skipna)
        else:
            quantiles = data.quantile(
                q=q,
                dim=dim,
                interpolation=interpolation,
                keep_attrs=keep_attrs,
                numeric_only=numeric_only,
                skipna=skipna,
            )

        if add_rank_coord:
            quantiles = quantiles.assign_coords(rank=("quantile", rank))

    elif isinstance(data, pd.DataFrame):
        if dim is None or isinstance(dim, str):
            dim = 0
        # Ensure `q` is a list so we always have an index
        if isinstance(q, (float, int)):
            q = [q]
        quantiles = data.quantile(q=q, axis=dim, numeric_only=numeric_only)
        if add_rank_coord:
            quantiles["rank"] = rank

    elif isinstance(data, pd.Series):
        if dim is None or isinstance(dim, str):
            dim = 0
        if add_rank_coord:
            # Ensure `q` is a list so we always have an index
            if isinstance(q, (float, int)):
                q = [q]
            quantiles = data.quantile(q=q, interpolation=interpolation)
            # Transform the Series to a DataFrame so we can have a rank column
            if quantiles.name is None:
                quantiles.name = "value"
            quantiles = quantiles.to_frame()
            quantiles["rank"] = rank
        else:
            quantiles = data.quantile(q=q, interpolation=interpolation)

    return quantiles


def rank(
    data: T.Union[xr.Dataset, xr.DataArray] = None, dim: str = "time", percent: bool = False, starting_rank: int = 1
) -> T.Union[xr.Dataset, xr.DataArray]:
    """
    Ranks data over the selected dimension. Equal values are assigned a rank
    that is the average of the ranks that would have been otherwise assigned
    to all of the values within that set. Replicates `xr.DataArray.rank` but
    with support for dask arrays - xarray documentation:
    http://xarray.pydata.org/en/stable/generated/xarray.DataArray.rank.html

    Can optionally set the starting rank to begin at 0 instead of the default of 1
    (the default from the `bottleneck` ranking function). NaNs in the input array
    are returned as NaNs.

    Args:
        data: Input data
        dim: Dimension over which to compute the rank
        percent: Optional. If `True`, compute percentage ranks, otherwise compute integer ranks.
        starting_rank: Starting number for the ranks if computing integer ranks.

    Returns:
        Ranked data with the same coordinates.
    """

    def _nan_rank(data):
        ranked = bn.nanrankdata(data, axis=-1)
        if percent:
            # Divide by the number of non-NaN values
            count = np.count_nonzero(~np.isnan(data), axis=-1, keepdims=True)
            ranked = ranked / count
        return ranked

    # Check if data is stored as Dask array, and if so, re-chunk over the ranking dimension
    if data.chunks is not None:
        if len(data.chunks) > 0:
            data = data.chunk({dim: -1})

    ranked_data = xr.apply_ufunc(
        _nan_rank,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[np.float32],
    )

    if percent:
        return ranked_data
    else:
        if starting_rank != 1:
            # By default rank starts at 1, so if choosing another
            # starting number, we need to add it and subtract 1
            ranked_data = ranked_data - 1 + starting_rank
        # As rank values are integers, cast to `int` dtype
        return ranked_data.astype(int)
