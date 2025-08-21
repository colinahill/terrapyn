import calendar
import datetime as dt

import ee
import geemap

from terrapyn.logger import logger


def add_aggregated_bands(
	img: ee.Image | ee.ImageCollection, bands: list, aggregation: str = "mean", name: str = None
) -> ee.Image | ee.ImageCollection:
	"""
	Aggregate bands and add to original ee.Image / Collection, optionally renaming the band

	Args:
		img: ee.Image or ee.ImageCollection
		bands: list of bands to aggregate
		aggregation: aggregation function to apply (e.g. "mean", "sum", "median")
		name: name of the new band

	Returns:
		ee.Image or ee.ImageCollection with aggregated bands added
	"""
	if name is None:
		name = aggregation + "_" + "_".join(bands)
	if isinstance(img, ee.ImageCollection):
		return img.map(lambda img: img.addBands(img.select(bands).reduce(aggregation).rename(name)))
	elif isinstance(img, ee.Image):
		return img.addBands(img.select(bands).reduce(aggregation).rename(name))
	else:
		raise TypeError("`img` must be an ee.ImageCollection or ee.Image")


def scale_monthly_values(img_col: ee.ImageCollection) -> ee.ImageCollection:
	"""
	Multiply monthly values by number of days in month.

	Args:
		img_col: ee.ImageCollection

	Returns:
		ee.ImageCollection
	"""
	dates = [dt.datetime.fromisoformat(date) for date in geemap.image_dates(img_col).getInfo()]
	days_in_month = ee.List([calendar.monthrange(date.year, date.month)[1] for date in dates])

	def _scale(img_scalefactor_list):
		img_scalefactor_list = ee.List(img_scalefactor_list)
		img = ee.Image(img_scalefactor_list.get(0))
		scale = img_scalefactor_list.getNumber(1)
		return img.multiply(scale).copyProperties(img, ["system:time_start"])

	# Map the scaling function over the list to scale each image
	zipped_list = img_col.toList(len(dates)).zip(days_in_month)
	return ee.ImageCollection.fromImages(zipped_list.map(_scale))


def scale_image_bands(
	img: ee.Image, band_names: list[str] = None, weight_dict: dict = None, default_weight: float = 1
) -> ee.Image:
	"""
	Multiply each band in an ee.Image by a dictionary of weights, where the key is the band name
	and the value is the weight.

	Args:
		img: ee.Image
		band_names: List of band names to scale. If None, scale all bands in the image.
		weight_dict: Dictionary of band names and weights
		default_weight: Default weight to apply if band name not in `weight_dict`

	Returns:
		ee.Image
	"""
	if band_names is None:
		band_names = img.bandNames().getInfo()

	# check all dict items are in image bands
	diff = set(weight_dict.keys()).difference(set(band_names))
	if diff:
		logger.warning(f"Some values in `weight_dict` are not an image band name and will be ignored: {diff}")

	# match weight values to image band order
	weights = [weight_dict.get(band, default_weight) for band in band_names]

	return img.multiply(ee.Image(weights))


def yearly_aggregation(img_col: ee.ImageCollection, reducer="sum") -> ee.ImageCollection:
	"""
	Group ee.ImageCollection by year and apply reducer to values

	Args:
		img_col: ee.ImageCollection
		reducer: reducer function (e.g. "sum", "mean", "median")

	Returns:
		ee.ImageCollection
	"""
	dates = [dt.datetime.fromisoformat(date) for date in geemap.image_dates(img_col).getInfo()]
	start_year, end_year = dates[0].year, dates[-1].year
	years = ee.List.sequence(start_year, end_year)

	def _yearly_reducer(fc, year):
		return fc.filter(ee.Filter.calendarRange(year, year, "year")).reduce(reducer).set("year", year)

	return ee.ImageCollection.fromImages(years.map(lambda year: _yearly_reducer(img_col, year)))


def add_area_to_feature(feature, error_margin=1):
	"""Computes the feature's geometry area and add it as a property"""
	error = ee.ErrorMargin(error_margin)
	area = ee.Number(feature.geometry(error).area(error))
	return feature.set({"area_m2": area})


def mask_exclude_geometry(img: ee.Image, mask: ee.Geometry | ee.Feature | ee.FeatureCollection) -> ee.Image:
	"""
	Set a mask on an image that excludes the geometry in `mask`.

	Args:
		img: An ee.Image
		mask: The geometry(s) to mask out

	Returns:
		An ee.Image
	"""
	img = img.updateMask(ee.Image(1).toByte().paint(mask, 0))
	return img


def mask_include(img: ee.Image, mask: ee.Geometry | ee.Feature | ee.FeatureCollection) -> ee.Image:
	"""
	Set a mask on an image to include regions in `mask`.

	Args:
		img: An ee.Image
		mask: The geometry(s) to mask in

	Returns:
		An ee.Image
	"""
	img = img.mask(ee.Image(0).toByte().paint(mask, 1))
	return img


def point_to_square(ee_point: ee.Geometry.Point, length: float = 10) -> ee.Geometry.Polygon:
	"""
	Convert a point to a square polygon with a side length of `length` meters

	Args:
		ee_point: An ee.Geometry.Point
		length: Length in meters

	Returns:
		ee.Geometry.Polygon
	"""
	return ee_point.buffer(length / 2.0).bounds()
