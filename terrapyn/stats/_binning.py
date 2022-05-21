import typing as T

import numpy as np
import pandas as pd
import xarray as xr


def _return_sorted_array(values: T.Union[float, int, np.ndarray, T.List]) -> np.ndarray:
    """ Sort values or ensure value is an array """
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
