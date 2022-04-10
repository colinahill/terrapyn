"""Statistical and mathematical operations"""

import typing as T

import numpy as np
import pandas as pd


def nearest_multiple(x, base: float = 0.5) -> np.ndarray:
    """
    Round a value to the nearest multiple of a given base. Rounding follows `numpy.round`,
    so 0.5 is closer to 1 than 0. Works for single values or numpy.arrays or lists.

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
