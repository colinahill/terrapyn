import typing as T

import bottleneck as bn
import numpy as np
import pandas as pd
import xarray as xr


def _return_sorted_array(values: T.Union[float, int, np.ndarray, T.List]) -> np.ndarray:
    """Sort values or ensure value is an array"""
    if isinstance(values, (T.List, np.ndarray)):
        return np.sort(values)
    else:
        return np.array([values])


def digitize(
    data: T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, np.ndarray, T.List] = None,  # noqa: C901
    bins: T.Union[float, int, np.ndarray, T.List] = [5, 10],
    closed_right: bool = False,
    columns: T.Union[str, np.ndarray, T.List] = None,
    keep_attrs: bool = True,
) -> T.Union[xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, np.ndarray, T.List]:
    """
    Return the indices of the bins to which each value in the input data belongs. This function is essentially a
    wrapper around numpy.digitize, and provides consistent usage for a variety of data structures.
    Automatically sorts the bins from lowest to highest. Accepts single values for bins or an array/list.

    ==============  ============================
    `closed_right`  returned index `i` satisfies
    ==============  ============================
    ``False``       ``bins[i-1] <= x < bins[i]``
    ``True``        ``bins[i-1] < x <= bins[i]``
    ==============  ============================

    If values are beyond the bounds of `bins`, 0 or ``len(bins)`` is returned as appropriate.

    A potential use case is for labelling values for scoring using a confusion matrix.

    Args:
        data: Input data
        bins: Array/list of values to use for thresholds.
        closed_right: Indicating whether the intervals include the right or the left bin edge.
        By default, closed_right=False, meaning that the interval does not include the right edge.
        columns: List of column names for which to apply the thresholding. Only used for pandas.Dataframe type.
        keep_attrs: Keep attributes if data is of type xr.DataArray. Ignored for other types of data.

    Returns:
        digitized_data: The data (same type as input) with values replaced by integers corresponding
        to indices of the given bins.

    >>> array = np.arange(1, 8, 0.5)
    >>> array
    array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
           7.5])
    >>> digitize(array, bins=[3, 6], closed_right=False)
    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
    >>> digitize(array, bins=[6, 3], closed_right=True)
    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2])
    >>> digitize(array, bins=3)
    array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    """
    bins = _return_sorted_array(bins)

    if isinstance(data, (np.ndarray, T.List)):
        digitized_data = np.digitize(data, bins=bins, right=closed_right)

    elif isinstance(data, pd.Series):
        result = np.digitize(data, bins=bins, right=True)
        digitized_data = pd.Series(result, index=data.index)

    elif isinstance(data, pd.DataFrame):
        if columns is None:
            dataframe = data
        else:
            # If column names(s) are given, ensure names are a list so a dataframe is always created
            if isinstance(columns, str):
                columns = [columns]
            dataframe = data[columns]

        # Apply digitize to dataframe
        result = dataframe.apply(np.digitize, axis=1, result_type="expand", bins=bins, right=closed_right)
        result.index = data.index
        if columns is None:
            result.columns = data.columns
        else:
            result.columns = columns
        digitized_data = result

    elif isinstance(data, (xr.DataArray, xr.Dataset)):
        digitized_data = xr.apply_ufunc(
            np.digitize,
            data,
            kwargs={"bins": bins, "right": closed_right},
            keep_attrs=keep_attrs,
            dask="allowed",
            output_dtypes=["i8"],
        )
    else:
        allowed_types = ", ".join(
            "{}".format(obj) for obj in [xr.Dataset, xr.DataArray, pd.DataFrame, pd.Series, np.ndarray, list]
        )
        raise TypeError("'data' is of type {}, but must be one of type {}.".format(type(data), allowed_types))
    return digitized_data


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
            quantiles = data.quantile(q=q, dim=dim, method=interpolation, keep_attrs=keep_attrs, skipna=skipna)
        else:
            quantiles = data.quantile(
                q=q,
                dim=dim,
                method=interpolation,
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


def nearest_multiple(x, base: float = 0.5) -> np.ndarray:
    """
    Round a value to the nearest multiple of a given base. Rounding follows `numpy.round`,
    so 0.5 is closer to 1 than 0. Works for single values or arrays/lists.

    Args:
        x: Value/array
        base: Base used to round the value.

    Returns:
        Rounded value(s)

    :Example:
    >>> nearest_multiple(6.73, base=0.02)
    6.72
    >>> nearest_multiple([1.77, 1.771], base=0.02)
    array([1.76, 1.78])
    """
    x = np.float64(x)
    base = np.float64(base)
    return np.round(np.divide(x, base), decimals=0) * base


def sigma_clip(
    data: T.Union[np.ndarray, pd.Series] = None,
    low_sigma: T.Union[int, float] = 7,
    upp_sigma: T.Union[int, float] = 7,
    n_iter: int = None,
    return_flags: bool = True,
    return_thresholds: bool = False,
) -> T.Union[
    T.Union[np.ndarray, pd.Series], T.Tuple[T.Union[np.ndarray, pd.Series], T.Optional[float], T.Optional[float]]
]:
    r"""
    Calculates the mean and standard deviation (`std`) of an array, and detemines if the values lie outside
    the range :math:`(mean - low_sigma * std)` to :math:`(mean + upp_sigma * std)`. `NaN` values are always
    flagged as bad points. If `return_flags==True`, returns an array mask where `True` indicates the value
    is rejected/bad (lying outside the defined range).

    If `n_iter` is `None`, perform iterative sigma-clipping of array elements. Starting from the full sample,
    all elements outside the critical range are removed, i.e. until all elements of `data` satisfy both conditions::

        data < mean(data) - std(data) * low_sigma
        data > mean(data) + std(data) * upp_sigma

    The iteration continues with the updated sample until no elements are outside the (updated) range.
    If `n_iter` is given, the iterative process is repeated `n_iter` times.

    Args:
        data: Input data
        low_sigma: Number of standard deviations to use as the lower rejection threshold
        upp_sigma: Number of standard deviations to use as the upper rejection threshold
        n_iter: If `None`, perform iterative sigma-clipping of array elements until no elements
        are outside the (updated) critical range. Otherwise, if `n_iter` is given, repeate the
        iterative process `n_iter` times.
        return_flags: If `True`, return a boolean array mask where `True` indicates the value is
        outside the defined range. If `False`, return the input array with clipped elements removed
        return_thresholds: If `True`, return the lower and upper threshold values used for clipping.

    Returns:
        If `return_flags==True`, an array mask for the values, where `True` indicates the value is outside
        the defined range. If `return_flags==False`, the input array with clipped elements removed.
        if `return_thresholds==True`, additionally return the lower and upper threshold values used for clipping.
    """
    if n_iter is not None and n_iter < 1:
        raise ValueError("`n_iter` must be `None` or `>=1`")

    # Ensure data is a 1-D array
    if isinstance(data, pd.Series):
        array = data.values
    elif isinstance(data, np.ndarray):
        array = np.asarray(data).ravel()
    else:
        raise TypeError("`data` must by of type `pd.Series` or `np.ndarray`")

    if len(array.shape) > 1:
        raise ValueError("'data' must be 1-D")

    # Initialize starting mask, including only non-NaN points
    mask = np.isfinite(array)

    iteration = 0
    while True:
        iteration += 1

        # Mean and std of subset of elements, where `mask` are good points
        mean = array[mask].mean()
        std = array[mask].std()
        lower = mean - low_sigma * std
        upper = mean + upp_sigma * std

        # Size of subset
        size = array[mask].size

        # mask_temp is boolean mask for good points in whole array based on new rejection criteria
        mask_temp = (array >= lower) & (array <= upper)

        # Update mask with temporary mask
        mask = mask_temp

        # check to see if the size of the array is equal to the number of included elements. If True, nothing more to do
        # OR check if the required number of iterations has been reached
        if (size - np.sum(mask)) == 0 or (iteration == n_iter):
            break

    if return_flags:
        # `mask` is `True` for good points, so invert mask so `True` is a rejected point
        mask = np.logical_not(mask)
        if return_thresholds:
            return mask, lower, upper
        else:
            return mask
    else:
        if isinstance(data, pd.Series):
            array = data[mask]
        else:
            array = array[mask]

        if return_thresholds:
            return array, lower, upper
        else:
            return array


def normalize_weights(weights: T.Iterable) -> T.List:
    """
    Normalize a list of weights so they sum to 1
    """
    return [i / sum(weights) for i in weights]
