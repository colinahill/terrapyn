import ee
import shapely

# from polygon_geohasher.polygon_geohasher import geohash_to_polygon
import terrapyn as tp


def ee_geom_to_shapely(ee_geometry: ee.Geometry) -> shapely.Geometry:
	"""
	Convert Earth Engine Geometry to Shapely Geometry.

	Args:
		ee_geometry: Earth Engine Geometry

	Returns:
		Shapely Geometry
	"""
	return tp.io.geometry_to_shapely(ee_geom_to_geojson(ee_geometry))


def ee_geom_to_geojson(ee_geometry, safe=True):
	"""
	Convert ee geometry to geojson. If `safe==True` this converts to shapely
	and back to fix issues converting between ee -> geojson
	"""
	geometry_geojson = ee_geometry.getInfo()
	if safe:
		geometry_shapely = tp.io.geometry_to_shapely(geometry_geojson)
		return tp.io.shapely_to_geojson(geometry_shapely)
	return geometry_geojson


def shapely_to_ee(shapely_polygon: shapely.Polygon) -> ee.Geometry.Polygon:
	"""
	Convert a shapely polygon to an ee.Geometry.Polygon

	Args:
		shapely_polygon: Shapely Polygon

	Returns:
		Earth Engine Geometry
	"""
	lons, lats = shapely_polygon.exterior.coords.xy
	lons_lats = [list(item) for item in list(zip(lons, lats, strict=False))]
	return ee.Geometry.Polygon([lons_lats])


# def geohash_to_ee_polygon(geohash: str) -> ee.Geometry.Polygon:
# 	"""Convert a geohash to an ee.Geometry.Polygon"""
# 	shapely_polygon = geohash_to_polygon(geohash)
# 	ee_polygon = shapely_to_ee(shapely_polygon)
# 	return ee_polygon
