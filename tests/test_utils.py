import unittest

import numpy as np
import pandas as pd

import terrapyn as tp


class TestPandasToGeoPandas(unittest.TestCase):
	def test_geometry(self):
		df = pd.DataFrame({"lon": [1, 2], "lat": [3, 4], "value": [5, 6]})
		result = tp.utils.pandas_to_geopandas(df)
		self.assertEqual([(point.x, point.y) for point in result["geometry"].to_numpy()], [(1.0, 3.0), (2.0, 4.0)])


class TestSetDimValuesInData(unittest.TestCase):
	df = pd.DataFrame(
		{
			"time": pd.DatetimeIndex(
				["2019-03-15", "2019-03-16", "2019-03-17"], dtype="datetime64[ns]", name="time", freq=None
			),
			"id": [123, 456, 789],
			"val": [1, 3, 5],
		}
	).set_index("time")

	def test_replace_dataframe_index(self):
		expected = pd.DatetimeIndex(
			["2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[ns]", name="time", freq=None
		)
		result = tp.utils.set_dim_values_in_data(self.df, expected, dim="time")
		pd.testing.assert_index_equal(result.index, expected)

	def test_replace_dataframe_multiindex(self):
		expected = pd.DatetimeIndex(
			["2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[ns]", name="time", freq=None
		)
		result = tp.utils.set_dim_values_in_data(
			self.df.reset_index(drop=False).set_index(["time", "id"]), expected, dim="time"
		)
		pd.testing.assert_index_equal(result.index.get_level_values("time"), expected)

	def test_replace_dataframe_column(self):
		expected = [6, 7, 8]
		result = tp.utils.set_dim_values_in_data(self.df, expected, dim="val")
		self.assertEqual(result["val"].to_list(), expected)

	def test_replace_series_index(self):
		expected = pd.DatetimeIndex(
			["2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[ns]", name="time", freq=None
		)
		result = tp.utils.set_dim_values_in_data(self.df["val"], expected, dim="time")
		pd.testing.assert_index_equal(result.index, expected)

	def test_replace_series_values(self):
		expected = [6, 7, 8]
		result = tp.utils.set_dim_values_in_data(self.df["val"], expected)
		self.assertEqual(result.to_list(), expected)

	def test_replace_dataset_dimension(self):
		expected = pd.DatetimeIndex(
			["2011-01-01", "2011-01-02", "2011-01-03"], dtype="datetime64[ns]", name="time", freq=None
		)
		result = tp.utils.set_dim_values_in_data(self.df.to_xarray(), expected, dim="time")
		np.testing.assert_array_equal(result["time"].values, expected.values)

	def test_wrong_data_type(self):
		with self.assertRaises(TypeError):
			tp.utils.set_dim_values_in_data([1, 2, 3], [4, 5, 6])


class TestRemoveListElements(unittest.TestCase):
	input_list = ["item", 5, "foo", 3.14, True]

	def test_remove_element(self):
		result = tp.utils.remove_list_elements(self.input_list, ["item"])
		self.assertEqual(result, [5, "foo", 3.14, True])

	def test_missing_list(self):
		with self.assertRaises(ValueError):
			tp.utils.remove_list_elements(self.input_list, None)

	def test_string(self):
		result = tp.utils.remove_list_elements(self.input_list, "item")
		self.assertEqual(result, [5, "foo", 3.14, True])
