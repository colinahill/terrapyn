import typing as T

import xarray as xr

import terrapyn as tp


def chunk_xarray(
    data: T.Union[xr.Dataset, xr.DataArray] = None,
    coords_no_chunking: T.Union[str, T.Iterable[str]] = None,
    coords_chunking: T.Dict = None,
) -> T.Union[xr.Dataset, xr.DataArray]:
    """
    Chunks xarrary data structures. If coordinate names are not given in `coords_no_chunking`, their chunk sizes
    are automatically determined by Dask. Otherwise, they can be set explicitly with `coords_chunking`

    Args:
        coords_no_chunking: List of coordinates that will have no chunking along this direction
        coords_chunking: (Optional) Dictionary of {coordinate name: chunk size} that will set the
        chunk size for those coordinates.

    Returns:
        Xarray data that has been chunked
    """
    # List of all coords in data
    coords = list(data.coords)

    # create chunks dict
    chunks = {}

    if coords_no_chunking is not None:
        # ensure coords_no_chunking is a list
        coords_no_chunking = tp.utils.ensure_list(coords_no_chunking)

        # Generate dict with coords that will not be chunked, where -1 means no chunking along this dimension
        chunks.update({i: -1 for i in coords_no_chunking})

        # Remove non-chunked coords from list of coords
        coords = tp.utils.remove_list_elements(coords, coords_no_chunking)

    if coords_chunking is not None:
        # combine provided chunk sizes with existing chunks
        chunks.update(coords_chunking)

        # Remove these coords from list of coords
        coords = tp.utils.remove_list_elements(coords, coords_chunking.keys())

    # Finally, set chunk sizes to 'auto' for all remaining coords
    if len(coords) > 0:
        chunks.update({i: "auto" for i in coords})

    return data.chunk(chunks)
