import unittest

import xarray as xr

import terrapyn as tp
from terrapyn import TEST_DATA_DIR


class TestChunkXarray(unittest.TestCase):
	ds = xr.open_dataset(TEST_DATA_DIR / "lat_10_lon_10_time_10_D_test_data.nc")

	def test_dataset_chunk_size_provided(self):
		result = tp.dask_utils.chunk_xarray(self.ds, "time", {"lat": 3, "lon": 3}).chunks
		expected = {"lat": (3, 3, 3, 1), "lon": (3, 3, 3, 1), "time": (10,)}
		self.assertEqual(result, expected)

	def test_dataset_all_auto_chunked(self):
		result = tp.dask_utils.chunk_xarray(self.ds).chunks
		expected = {"lat": (10,), "lon": (10,), "time": (10,)}
		self.assertEqual(result, expected)
