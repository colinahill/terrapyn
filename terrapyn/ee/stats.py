import typing as T

import ee

import terrapyn as tp


def image_percentiles(
	img: ee.Image,
	geometry: ee.Geometry,
	percentiles: list[int] | None = None,
	scale: float = 5000,
	**kwargs,
) -> dict:
	"""
	Calculate percentiles in a region for an ee.Image

	Args:
		img: ee.Image
		geometry: ee.Geometry
		percentiles: List of percentiles to calculate. Must be integers between 0 and 100.
		Default is [10, 20, 30, 40, 50, 60, 70, 80, 90, 100].
		scale: Image scale to compute the percentiles.
		kwargs: Additional arguments to pass to ee.Image.reduceRegion()

	Returns:
		Dictionary of `{percentile: percentile value}`
	"""
	# Check is an ee.Image
	if not isinstance(img, ee.Image):
		raise ValueError("img must be an ee.Image")
	if percentiles is None:
		percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
	percentile_labels = [f"p{i:02d}" for i in percentiles]
	pct = img.rename("img").reduceRegion(
		reducer=ee.Reducer.percentile(percentiles=percentiles, outputNames=percentile_labels),
		geometry=geometry,
		scale=scale,
		**kwargs,
	)
	results = pct.getInfo()

	# rename keys according to label order
	return dict((label, results[f"img_{label}"]) for label in percentile_labels)


def mean_center_image(img: ee.Image, geometry: ee.Geometry, scale: float = 250) -> ee.Image:
	"""
	Mean-center an ee.Image by calculating the mean within the geometry and subtracting it from the pixel image values.

	Args:
		img: ee.Image
		geometry: ee.Geometry
		scale: Image scale to compute the mean.

	Returns:
		ee.Image
	"""
	band_names = img.bandNames()
	mean_dict = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geometry, scale=scale, maxPixels=1e12)
	means = ee.Image.constant(mean_dict.values(band_names))
	centered = img.subtract(means)
	return centered


def image_min_max(img: ee.Image, geometry: ee.Geometry, scale: float = 250, **kwargs):
	"""
	Compute the minimum and maximum of each band in an ee.Image and return as dictionary.

	Args:
		img: ee.Image
		geometry: Region in which to compute the min/max.
		scale: Image scale to compute the min/max.
		kwargs: Additional arguments to pass to ee.Image.reduceRegion()

	Returns:
		Dictionary of `{band: {"min": min_value, "max": max_value}}`
	"""
	min_max = img.reduceRegion(reducer=ee.Reducer.minMax(), geometry=geometry, scale=scale, maxPixels=1e13, **kwargs)
	return min_max.getInfo()


def bin_values(img: ee.Image, bins: list) -> ee.Image:
	"""
	Bin values in an ee.Image, where the pixel label is the index in `bins` where the value is greater
	than that bin but less than the next bin.

	Psuedo-example using lists:
		bins = [0, 4, 6, 9]
		img = [1, 3, 5, 10]
		bin_values(img, bins)
		[0, 0, 1, 3]

	Args:
		img: ee.Image
		bins: list of bin thresholds

	Returns:
		ee.Image
	"""
	thresholds = ee.Image(bins)
	return img.gt(thresholds).reduce("sum").subtract(1).rename("class")


def bin_and_remap(img: ee.Image, bins: list, labels: list, default_value=None) -> ee.Image:
	"""
	Bin values in an image and apply a label, where the label corresponds to where the value is greater than the
	bin but less than the next bin. If no mapping exists, the default_value is used.

	Pseudo example using lists:
		bins = [0, 4, 6, 9]
		img = [1, 3, 5, 10]
		labels = [111, 222, 333, 444]
		bin_and_remap(img, bins, labels)
		[111, 111, 222, 444]

	Args:
		img: ee.Image
		bins: list of bin thresholds
		labels: list of labels to apply
		default_value: value to use if no mapping exists

	Returns:
		ee.Image

	"""
	if len(bins) != len(labels):
		raise ValueError("bins and labels must be the same length")

	class_index = bin_values(img, bins)
	indices = [i for i in range(len(bins))]
	return class_index.remap(indices, labels, defaultValue=default_value)


def principal_components(img: ee.Image, geometry: ee.Geometry, scale: float, **kwargs):
	"""
	Computes Principal Components of and image in the given region at the given scale.

	Args:
		img: ee.Image
		geometry: ee.Geometry
		scale: Image scale to use to compute the principal components.
		kwargs: Additional arguments to pass to ee.Image.reduceRegion()

	Returns:
		ee.Image
	"""

	def _get_principal_component_band_names(prefix="pc_", n_pc=3):
		"""Returns a list of new Principal Component band names"""
		seq = ee.List.sequence(1, n_pc)
		return seq.map(lambda n: ee.String(prefix).cat(ee.Number(n).int().format()))

	centered = mean_center_image(img, geometry, scale)

	# number of principal components
	n_pc = centered.bandNames().length()

	# Collapse the bands of the image into a 1D array per pixel
	arrays = centered.toArray()

	# Compute the covariance of the bands within the region
	covar = arrays.reduceRegion(
		reducer=ee.Reducer.centeredCovariance(), geometry=geometry, scale=scale, maxPixels=1e13, **kwargs
	)

	# Get the 'array' covariance result and cast to an array.
	# This represents the band-to-band covariance within the region.
	covar_array = ee.Array(covar.get("array"))

	# Perform an eigen analysis and slice apart the values and vectors
	eigens = covar_array.eigen()

	# This is a P-length vector of Eigenvalues
	eigen_values = eigens.slice(1, 0, 1)

	# This is a PxP matrix with eigenvectors in rows
	eigen_vectors = eigens.slice(1, 1)

	# Convert the array image to 2D arrays for matrix computations
	array_image = arrays.toArray(1)

	# Left multiply the image array by the matrix of eigenvectors
	principal_components = ee.Image(eigen_vectors).matrixMultiply(array_image)

	# Turn the square roots of the Eigenvalues into a P-band image
	sd_image = (
		ee.Image(eigen_values.sqrt()).arrayProject([0]).arrayFlatten([_get_principal_component_band_names("sd_", n_pc)])
	)

	# Turn the PCs into a P-band image, normalized by SD
	pc_image = (
		principal_components
		# Throw out an an unneeded dimension, [[]] -> []
		.arrayProject([0])
		# Make the one band array image a multi-band image, [] -> image
		.arrayFlatten([_get_principal_component_band_names("pc_", n_pc)])
		# Normalize the PCs by their SDs
		.divide(sd_image)
	)

	return pc_image


def class_count(img: ee.Image, geometry: ee.Geometry, scale: float = 1000, **kwargs) -> dict:
	"""
	Count frequency of classes in a single-band ee.Image.

	Args:
		img: ee.Image
		geometry: ee.Geometry
		scale: Image scale to compute the class counts.
		kwargs: Additional arguments to pass to ee.Image.reduceRegion()

	Returns:
		Dictionary of {class: count}
	"""
	reduction = img.reduceRegion(
		ee.Reducer.frequencyHistogram().unweighted(),
		geometry=geometry,
		scale=scale,
		maxPixels=1e13,
		**kwargs,
	)
	return reduction.get(img.bandNames().get(0)).getInfo()


def weighted_mean(img, weight_dict, output_bandname="mean"):
	"""
	Return the weighted mean of all bands in an image, where weights are specified by a dictionary.
	Weights will automatically be normalized.
	"""
	bands, weights = list(weight_dict.keys()), list(weight_dict.values())

	# Normalize weights so they sum to 1
	weights = tp.stats.normalize_weights(weights)

	weights_dict = dict(zip(bands, weights, strict=False))
	return tp.ee.utils.scale_image_bands(img, weight_dict=weights_dict).reduce("sum").rename(output_bandname)


def _layer_weights(layer_tops: T.Iterable, layer_bottoms: T.Iterable, total_depth: float = 30) -> list:
	"""
	Calculate a weighting for each layer based on thickness and range overlap, typically used for soil horizons
	"""
	weights = []
	for top, bottom in zip(layer_tops, layer_bottoms, strict=False):
		overlap = min(total_depth, bottom) - max(0, top)
		# if depth range doesn't overlap, weight = 0
		weights.append(overlap if overlap > 0 else 0)
	return weights
