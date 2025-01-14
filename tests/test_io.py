import geopandas as gpd
import shapely

import terrapyn as tp
from terrapyn import TEST_DATA_DIR


def test_kml_to_geodataframe():
	kml_filepath = TEST_DATA_DIR / "KML_example.kml"
	df = tp.io.kml_to_geodataframe(kml_filepath)
	assert isinstance(df, gpd.GeoDataFrame)
	assert len(df) > 0
	assert "layer" in df.columns
	assert "geometry" in df.columns
	assert df.crs is not None


def test_kmz_to_geodataframe():
	kmz_filepath = TEST_DATA_DIR / "KMZ_example.kmz"
	df = tp.io.kmz_to_geodataframe(kmz_filepath)
	assert isinstance(df, gpd.GeoDataFrame)
	assert len(df) > 0
	assert "layer" in df.columns
	assert "geometry" in df.columns
	assert df.crs is not None


def test_geojson_to_single_shapely_geometry():
	geojson = {
		"type": "FeatureCollection",
		"features": [
			{
				"type": "Feature",
				"properties": {},
				"geometry": {
					"type": "Polygon",
					"coordinates": [
						[
							[-104.05, 48.99],
							[-97.22, 48.98],
							[-96.58, 45.94],
							[-104.03, 45.94],
							[-104.05, 48.99],
						]
					],
				},
			},
			{
				"type": "Feature",
				"properties": {},
				"geometry": {
					"type": "Polygon",
					"coordinates": [
						[
							[-109.05, 41.00],
							[-102.06, 40.99],
							[-102.03, 36.99],
							[-109.04, 36.99],
							[-109.05, 41.00],
						]
					],
				},
			},
		],
	}

	result = tp.io.geojson_to_single_shapely_geometry(geojson)
	assert isinstance(result, shapely.MultiPolygon)
	assert result.is_valid


def test_geometry_to_shapely():
	geojson = {
		"type": "Polygon",
		"coordinates": [
			[
				[-104.05, 48.99],
				[-97.22, 48.98],
				[-96.58, 45.94],
				[-104.03, 45.94],
				[-104.05, 48.99],
			]
		],
	}

	result = tp.io.geometry_to_shapely(geojson)
	assert isinstance(result, shapely.Polygon)
	assert result.is_valid


def test_shapely_to_geojson():
	# Create a shapely shape object
	geometry_shapely = shapely.geometry.Polygon(
		[
			(0, 0),
			(1, 0),
			(1, 1),
			(0, 1),
			(0, 0),
		]
	)

	# Convert to geoJSON
	geojson = tp.io.shapely_to_geojson(geometry_shapely)
	assert isinstance(geojson, dict)
	assert geojson["type"] == "Polygon"
	assert geojson["coordinates"] == [
		[
			[0.0, 0.0],
			[1.0, 0.0],
			[1.0, 1.0],
			[0.0, 1.0],
			[0.0, 0.0],
		]
	]
