import ee


def reduce_regions(
	image_collection: ee.ImageCollection,
	feature_collection: ee.FeatureCollection,
	reducer: ee.Reducer = ee.Reducer.mean(),  # noqa: B008
	scale: float = 1_000,
	crs: str = "EPSG:4326",
	crsTransform: list = None,
	tileScale: int = 1,
	drop_na: bool = True,
	date_property: str = "system:time_start",
) -> ee.FeatureCollection:
	"""
	Reduce an image collection to a feature collection using a reducer, optionally dropping features with Null values.

	Args:
		image_collection: The image collection to reduce.
		feature_collection: The feature collection to use for reduction/extraction/
		reducer: The reducer to use.
		scale: The scale to use.
		crs: The CRS to use.
		crsTransform: The CRS transform to use.
		tileScale: The tile scale to use.
		drop_na: Whether or not to drop features with Null values.
		date_property: The property to use for the date.
	"""
	image_collection = image_collection.filterBounds(feature_collection)
	if image_collection.size().getInfo() == 0:
		return None

	band_names = image_collection.first().bandNames().getInfo()
	reducer_names = reducer.getOutputs().getInfo()

	if len(band_names) == 1:
		if len(reducer_names) == 1:
			reducer = reducer.setOutputs(band_names)
			output_names = band_names
		elif len(reducer_names) > 1:
			output_names = [f"{band}_{reducer}" for band in band_names for reducer in reducer_names]
			reducer = reducer.setOutputs(output_names)
	else:
		if len(reducer_names) == 1:
			output_names = band_names
		elif len(reducer_names) > 1:
			output_names = [f"{band}_{reducer}" for band in band_names for reducer in reducer_names]
			reducer = reducer.setOutputs(output_names)

	def img_reduce_regions(img):
		date = ee.Date(img.get(date_property))

		fc = img.reduceRegions(
			collection=feature_collection,
			reducer=reducer,
			scale=scale,
			crs=crs,
			crsTransform=crsTransform,
			tileScale=tileScale,
		)

		if drop_na:
			fc = fc.filter(ee.Filter.Or([ee.Filter.neq(key, None) for key in output_names]))

		return fc.map(lambda f: f.set({"date": date}))

	output_feature_collection = image_collection.map(img_reduce_regions).flatten()

	return output_feature_collection
