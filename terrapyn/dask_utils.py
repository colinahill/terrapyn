import typing as T

import dask
import dask.distributed
import xarray as xr

import terrapyn as tp


def create_cluster_and_client(n_workers: int = 4, threads_per_worker: int = 1, memory_limit: str = "1GB", **kwargs):
	"""Create a local cluster and initiate a client

	Args:
		n_workers: Number of workers to start.
		threads_per_worker: Number of threads per each worker.
		memory_limit: Memory limit per worker.

	Returns:
		Dask Client
	"""
	client = dask.distributed.Client(
		n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit, **kwargs
	)
	print(f"Client dashboard: {client.dashboard_link}")
	return client


def chunk_xarray(
	data: xr.Dataset | xr.DataArray = None,
	coords_no_chunking: str | T.Iterable[str] = None,
	coords_chunking: dict = None,
) -> xr.Dataset | xr.DataArray:
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


def uses_dask(data: xr.Dataset | xr.DataArray = None) -> xr.Dataset | xr.DataArray:
	"""Check if an xarray data structure uses Dask"""
	return (isinstance(data, xr.DataArray) and isinstance(data.data, dask.array.Array)) or (
		isinstance(data, xr.Dataset) and any(isinstance(var.data, dask.array.Array) for var in data.variables.values())
	)
