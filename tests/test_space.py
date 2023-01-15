from pathlib import Path
from unittest import TestCase

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

import terrapyn as tp

PACKAGE_ROOT_DIR = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = PACKAGE_ROOT_DIR / "tests" / "data"


class TestGetDataAtCoords(TestCase):

    np.random.seed(42)
    n_lat = 4
    n_lon = 4
    n_time = 3
    data = 5 + np.random.randn(n_time, n_lat, n_lon)
    da = xr.DataArray(
        data,
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2014-09-06", periods=n_time),
            "lat": 3 + np.arange(n_lat),
            "lon": 13 + np.arange(n_lon),
        },
        name="var",
    )
    ds = da.to_dataset()

    lats = [3.32, 4.67]
    lons = [15.02, 15.73]
    ids = ["a", "b"]

    def test_nearest_coordinate_values(self):
        df = tp.space.get_data_at_coords(self.ds, lats=self.lats, lons=self.lons)
        nearest_values_point_0 = df.loc[(slice(None), 0), "var"].values
        np.testing.assert_almost_equal(nearest_values_point_0, np.array([5.64768854, 4.09197592, 5.82254491]))
        nearest_values_point_1 = df.loc[(slice(None), 1), "var"].values
        np.testing.assert_almost_equal(nearest_values_point_1, np.array([4.53427025, 5.37569802, 4.6988963]))

    def test_missing_latitudes(self):
        self.assertRaises(
            ValueError,
            tp.space.get_data_at_coords,
            ds=self.ds,
            lats=None,
            lons=self.lons,
        )

    def test_linear_values(self):
        df = tp.space.get_data_at_coords(self.ds, lats=self.lats, lons=self.lons, method="linear")
        nearest_values_point_0 = df.loc[(slice(None), 0), "var"].values
        np.testing.assert_almost_equal(nearest_values_point_0, np.array([5.95248557, 4.38774388, 5.11628122]))
        nearest_values_point_1 = df.loc[(slice(None), 1), "var"].values
        np.testing.assert_almost_equal(nearest_values_point_1, np.array([5.01396221, 4.63833409, 4.7608919]))

    def test_id_names(self):
        df = tp.space.get_data_at_coords(self.ds, lats=self.lats, lons=self.lons, point_names=self.ids, method="linear")
        self.assertEqual(
            list(df.index.get_level_values("id")),
            ["a", "b", "a", "b", "a", "b"],
        )

    def test_point_names_dim(self):
        df = tp.space.get_data_at_coords(
            self.ds,
            lats=self.lats,
            lons=self.lons,
            point_names=self.ids,
            method="linear",
            point_names_dim="test",
        )
        self.assertEqual(list(df.index.names), ["time", "test"])

    def test_dataset_without_time(self):
        data = self.ds.isel(time=0).drop_vars("time")
        df = tp.space.get_data_at_coords(data, lats=self.lats, lons=self.lons, method="nearest")
        self.assertEqual(df.index.name, "id")
        np.testing.assert_almost_equal(df["var"].values, np.array([5.64768854, 4.53427025]))


class TestBBox(TestCase):

    min_lon = 10
    max_lon = 20
    min_lat = 30
    max_lat = 40
    shapely_rectangle = shapely.geometry.box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)
    shapely_polygon = shapely.geometry.polygon.Polygon([(0, 0), (1, 1), (2, 2), (2, 1), (0, 0)])

    def test_bounds(self):
        assert tp.space.BBox(geometry=self.shapely_rectangle).bounds == (10.0, 30.0, 20.0, 40.0)
        assert tp.space.BBox(geometry=self.shapely_polygon).bounds == (0.0, 0.0, 2.0, 2.0)

    def test_WNES(self):
        assert tp.space.BBox(
            min_lat=self.min_lat, max_lat=self.max_lat, min_lon=self.min_lon, max_lon=self.max_lon
        ).WNES == (self.min_lon, self.max_lat, self.max_lon, self.min_lat)

    def test_NWSE(self):
        assert tp.space.BBox(
            min_lat=self.min_lat, max_lat=self.max_lat, min_lon=self.min_lon, max_lon=self.max_lon
        ).NWSE == (self.max_lat, self.min_lon, self.min_lat, self.max_lon)

    def test_centroid(self):
        bbox = tp.space.BBox(min_lat=self.min_lat, max_lat=self.max_lat, min_lon=self.min_lon, max_lon=self.max_lon)
        assert isinstance(bbox.centroid, shapely.geometry.point.Point)
        self.assertEqual(bbox.centroid.coords[0], (15.0, 35.0))
        self.assertEqual(bbox.centroid_coords, (35.0, 15.0))

    def test_buffer(self):
        assert tp.space.BBox(geometry=self.shapely_polygon).buffer(1).bounds == (
            -0.99997179491729,
            -0.9997461638018406,
            3.0,
            3.0,
        )


class TestCropToBBox(TestCase):
    def test_pandas_dataframe(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2019-03-15", freq="1D", periods=3),
                "id": [123, 456, 789],
                "lat": [1, 3, 5],
                "lon": [10, 20, 30],
            }
        ).set_index(["time", "id"])
        bbox = tp.space.BBox(max_lat=4, min_lat=0, max_lon=25, min_lon=0)
        results = tp.space.crop_to_bbox(df, bbox=bbox).values
        expected = np.array([[1, 10], [3, 20]])
        np.testing.assert_equal(results, expected)

    def test_geopandas_geodataframe(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2019-03-15", freq="1D", periods=3),
                "id": [123, 456, 789],
                "lat": [1, 3, 5],
                "lon": [10, 20, 30],
            }
        ).set_index(["time", "id"])
        geometry = gpd.points_from_xy(df.lon, df.lat)
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        bbox = tp.space.BBox(max_lat=4, min_lat=0, max_lon=25, min_lon=0)
        results = tp.space.crop_to_bbox(gdf, bbox=bbox)[["lat", "lon"]].values
        expected = np.array([[1, 10], [3, 20]])
        np.testing.assert_equal(results, expected)

    def test_xarray_dataset(self):
        ds = xr.open_dataset(TEST_DATA_PATH / "lat_10_lon_10_time_10_D_test_data.nc")
        bbox = tp.space.BBox(min_lat=3.3, max_lat=5.7, min_lon=12.3, max_lon=14.4)
        results = tp.space.crop_to_bbox(ds, bbox=bbox)["var"].values
        expected = np.array(
            [
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
                [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            ]
        )
        np.testing.assert_almost_equal(results, expected)


class TestPointsInRadius(TestCase):

    df = pd.read_csv(TEST_DATA_PATH / "station_metadata.csv")

    def test_points_and_distance(self):
        result = tp.space.points_within_radius(self.df, point=[43.5, 1.2], radius_km=10, return_distance=True)
        np.testing.assert_almost_equal(
            result[["id", "distance_km"]].values,
            np.array([[7628099999, 6.8720606], [7629399999, 7.6216178]], dtype=object),
        )


class TestGenerateGrid(TestCase):
    def test_filled_dataset(self):
        ds = tp.space.generate_grid(fill_value=1, return_dataset=True, resolution=0.5)
        result = ", ".join(f"{i} {j}" for i, j in ds.dims.items())
        expected = "lat 361, lon 721"
        self.assertEqual(result, expected)
        self.assertEqual(ds["var"].isel(lat=10, lon=10).item(), 1)

    def test_dataarray_from_bbox(self):
        da = tp.space.generate_grid(bbox=tp.space.BBox(min_lat=41, max_lat=52, min_lon=-5.5, max_lon=9.5))
        np.testing.assert_almost_equal(
            da.lat.values, np.array([41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0])
        )
        np.testing.assert_almost_equal(
            da.lon.values,
            np.array([-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]),
        )


class TestBBoxFromData(TestCase):

    ds = xr.open_dataset(TEST_DATA_PATH / "lat_10_lon_10_time_10_D_test_data.nc")

    def test_dataset(self):
        bbox = tp.space.bbox_from_data(self.ds)
        self.assertEqual(bbox.NWSE, (9, 11, 0, 20))

    def test_dataframe(self):
        df = self.ds.to_dataframe()
        bbox = tp.space.bbox_from_data(df)
        self.assertEqual(bbox.NWSE, (9, 11, 0, 20))

    def test_series(self):
        series = self.ds.to_dataframe()["var"]
        bbox = tp.space.bbox_from_data(series)
        self.assertEqual(bbox.NWSE, (9, 11, 0, 20))


class TestMatchDataBBox(TestCase):

    input = xr.open_dataset(TEST_DATA_PATH / "lat_10_lon_10_time_10_D_test_data.nc")
    reference = input.isel(lat=slice(3, 7), lon=slice(3, 7))

    def test_dataset_and_dataset(self):
        result = tp.space.match_data_bbox(self.reference, self.input)
        np.testing.assert_equal(result.lat.values, np.array([3, 4, 5, 6]))
        np.testing.assert_equal(result.lon.values, np.array([14, 15, 16, 17]))

    def test_dataframe_and_dataset(self):
        df = self.reference.to_dataframe()
        df_new = tp.space.match_data_bbox(df, self.input)
        np.testing.assert_equal(df_new.lat.values, np.array([3, 4, 5, 6]))
        np.testing.assert_equal(df_new.lon.values, np.array([14, 15, 16, 17]))

        df = df.reset_index()
        df_new = tp.space.match_data_bbox(df, self.input)
        np.testing.assert_equal(df_new.lat.values, np.array([3, 4, 5, 6]))
        np.testing.assert_equal(df_new.lon.values, np.array([14, 15, 16, 17]))


class TestGetNearestPoint(TestCase):

    df = pd.read_csv(TEST_DATA_PATH / "station_metadata.csv")

    def test_single_nearest_point_geodesic_no_distance(self):
        result = tp.space.get_nearest_point(self.df, lats=43.44, lons=1.23, return_distance=False)
        np.testing.assert_equal(result[["lat", "lon"]].values, np.array([[43.45, 1.25]]))
        self.assertFalse("distance_km" in result.columns)

    def test_multiple_nearest_points_geodesic(self):
        result = tp.space.get_nearest_point(self.df, lats=[43.44, 42.1], lons=[1.23, 0.3])
        np.testing.assert_equal(result[["lat", "lon"]].values, np.array([[43.45, 1.25], [43.008, 1.103]]))
        np.testing.assert_almost_equal(result["distance_km"].values, np.array([1.96354701, 120.50944031]))

    def test_single_nearest_point_kdtree(self):
        result = tp.space.get_nearest_point(self.df, lats=43.44, lons=1.23, method="kdtree")
        np.testing.assert_equal(result[["lat", "lon"]].values, np.array([[43.45, 1.25]]))

    def test_multiple_nearest_points_kdtree(self):
        result = tp.space.get_nearest_point(self.df, lats=[43.44, 42.1], lons=[1.23, 0.3], method="kdtree")
        np.testing.assert_equal(result[["lat", "lon"]].values, np.array([[43.45, 1.25], [43.008, 1.103]]))


class TestGroupPointsByGrid(TestCase):

    df = pd.read_csv(TEST_DATA_PATH / "station_metadata.csv")

    def test_groups_with_cellsize(self):
        groups, lat_bounds, lon_bounds = tp.space.group_points_by_grid(self.df, cellsize=0.5)
        np.testing.assert_equal(
            groups.values,
            np.array(
                [list([7627099999]), list([7628099999, 7629399999]), list([7630099999, 7631099999])],
                dtype=object,
            ),
        )
        np.testing.assert_equal(lat_bounds, np.array([42.758, 43.258, 43.758, 44.258]))
        np.testing.assert_equal(lon_bounds, np.array([0.853, 1.353, 1.853]))

    def test_groups_with_arbitrary_bounds(self):
        lat_bounds = [42.5, 43.4, 43.6, 44]
        lon_bounds = [1, 1.3, 1.4]
        # cellsize will be ignored
        groups = tp.space.group_points_by_grid(
            self.df, cellsize=0.5, lat_bounds=lat_bounds, lon_bounds=lon_bounds, return_cell_bounds=False
        )
        np.testing.assert_equal(
            groups.values,
            np.array(
                [
                    list([7627099999]),
                    list([7628099999, 7629399999]),
                    list([7631099999]),
                    list([7630099999]),
                ],
                dtype=object,
            ),
        )
