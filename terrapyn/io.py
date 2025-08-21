import json
from pathlib import Path
from zipfile import ZipFile

import fiona
import geopandas as gpd
import pandas as pd
import shapely


def load_json(filepath: str | Path) -> dict:
	with open(filepath, "rb") as fp:
		return json.load(fp)


def _make_geom_2d(shapely_geometry: shapely.Geometry) -> shapely.Geometry:
	"""Converts a shapely geometry from 3-D to 2-D"""
	return shapely.wkb.loads(shapely.wkb.dumps(shapely_geometry, output_dimension=2))


def kml_to_geodataframe(filepath: Path = None) -> gpd.GeoDataFrame:
	"""
	Load a KML file and convert to a GeoPandas GeoDataFrame. Allows multiple folders/layers in KML.

	Args:
		filepath: Path to KML file

	Returns:
		GeoPandas GeoDataFrame
	"""
	filepath = Path(filepath)

	fiona.supported_drivers["KML"] = "rw"
	df = pd.DataFrame()
	for layer in fiona.listlayers(filepath):
		features = gpd.read_file(filepath, driver="KML", layer=layer)
		features["layer"] = layer
		df = pd.concat([df, features], ignore_index=True)

	# Convert geometry to 2-D
	df["geometry"] = df["geometry"].map(_make_geom_2d)
	return df


def kmz_to_geodataframe(filepath: Path = None) -> gpd.GeoDataFrame:
	"""
	Load a KMZ file and convert to a GeoPandas GeoDataFrame. Allows multiple folders/layers in KMZ.

	Args:
		filepath: Path to KMZ file

	Returns:
		GeoPandas GeoDataFrame
	"""
	filepath = Path(filepath)

	# Extract KML file from KMZ
	kmz = ZipFile(filepath, "r")
	kml_filepath = filepath.with_suffix(".kml")
	kmz.extract("doc.kml", kml_filepath)
	df = kml_to_geodataframe(kml_filepath / "doc.kml")

	return df


def geojson_to_single_shapely_geometry(geojson: dict) -> shapely.MultiPolygon | shapely.Polygon:
	"""
	Parse a geoJSON dictionary and convert and merge all features/geometries to a single (multi)polygon
	"""
	if not isinstance(geojson, dict):
		raise TypeError("geojson must be a dictionary type")

	if geojson["type"] == "GeometryCollection":
		# If geojson is a GeometryCOllection, convert to FeatureCollection
		features = [{"type": "Feature", "properties": {}, "geometry": geometry} for geometry in geojson["geometries"]]
		geojson = {"type": "FeatureCollection", "features": features}
	elif geojson["type"] == "Feature":
		# If geojson is a feature, convert to FeatureCollection
		geojson = {"type": "FeatureCollection", "features": [geojson]}
	elif geojson["type"] != "FeatureCollection":
		# Otherwise assume geojson is a geometry, convert to FeatureCollection
		geojson = {
			"type": "FeatureCollection",
			"features": [{"type": "Feature", "properties": {}, "geometry": geojson}],
		}

	gdf = gpd.GeoDataFrame.from_features(geojson)

	# Ensure all geometries are a Polygon or MultiPolygon. Point, MultiPoint, LineString, MultiLineString
	# are invalid as they do not define a region boundary.
	all_valid_geometry_types = gdf["geometry"].geom_type.isin({"Polygon", "MultiPolygon"}).all()

	if not all_valid_geometry_types:
		raise ValueError("All geoJSON features must be of type 'Polygon' or 'MultiPolygon'")

	# Dissolve all geometries to a single (multi)polygon
	shapely_geometry = gdf.dissolve()["geometry"].iloc[0]

	return shapely_geometry


def geometry_to_shapely(geometry: dict) -> shapely.Geometry:
	"""
	Convert a geoJSON geometry fragment to a shapely shape object

	Args:
		geometry: geoJSON geometry fragment

	Returns:
		Shapely shape object
	"""
	return shapely.geometry.shape(geometry)


def shapely_to_geojson(geometry_shapely: shapely.Geometry) -> dict:
	"""
	Convert a shapely shape object to a geoJSON

	Args:
		geometry_shapely: Shapely shape object

	Returns:
		geoJSON dictionary
	"""
	return json.loads(shapely.to_geojson(geometry_shapely))
