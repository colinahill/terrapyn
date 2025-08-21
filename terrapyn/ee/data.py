import ee

import terrapyn as tp


def soilgrids(
	params: list[str] | None = None,
	return_horizon_mean: bool = False,
	depth: int = 30,
):
	"""
	Return ee.Image of Soil Grids 250m v2.0 data, see https://gee-community-catalog.org/projects/isric/

	Data are converted to have units of:
	- bdod: kg/dm³
	- cec: cmol(c)/kg
	- cfvo: cmol(c)/kg
	- clay: %
	- sand: %
	- silt: %
	- nitrogen: %
	- phh2o: pH
	- soc: %
	- ocd: g/kg

	Args:
		params: List of parameters to return. Options are: 'bdod', 'cec', 'cfvo',
		'clay', 'sand', 'silt', 'nitrogen', 'phh2o', 'soc', 'ocd', 'ocs'. Default is all.
		return_horizon_mean: If True, return the weighted mean over the range 0 - `depth` [cm]

	Returns:
		ee.Image with bands of the requested parameters
	"""
	if params is None:
		params = ["bdod", "cec", "cfvo", "clay", "sand", "silt", "nitrogen", "phh2o", "soc", "ocd", "ocs"]

	param_dict = {
		"bdod": {
			"param": "bdod_mean",
			"description": "Bulk density of the fine earth fraction",
			"unit": "kg/dm³",
			"conversion_factor": 100,
		},
		"cec": {
			"param": "cec_mean",
			"description": "Cation Exchange Capacity of the soil",
			"unit": "cmol(c)/kg",
			"conversion_factor": 10,
		},
		"cfvo": {
			"param": "cfvo_mean",
			"description": "Volumetric fraction of coarse fragments (> 2 mm)",
			"unit": "cm3/100cm3 (vol%)",
			"conversion_factor": 10,
		},
		"clay": {
			"param": "clay_mean",
			"description": "Proportion of clay particles (< 0.002 mm) in the fine earth fraction",
			"unit": "g/100g (%)",
			"conversion_factor": 10,
		},
		"nitrogen": {
			"param": "nitrogen_mean",
			"description": "Total nitrogen (N)",
			"unit": "g/100g (%)",
			"conversion_factor": 1000,
		},
		"phh2o": {"param": "phh2o_mean", "description": "Soil pH", "unit": "pH", "conversion_factor": 10},
		"sand": {
			"param": "sand_mean",
			"description": "Proportion of sand particles (> 0.05 mm) in the fine earth fraction",
			"unit": "g/100g (%)",
			"conversion_factor": 10,
		},
		"silt": {
			"param": "silt_mean",
			"description": "Proportion of silt particles (≥ 0.002 mm and ≤ 0.05 mm) in the fine earth fraction",
			"unit": "g/100g (%)",
			"conversion_factor": 10,
		},
		"soc": {
			"param": "soc_mean",
			"description": "Soil organic carbon content in the fine earth fraction",
			"unit": "g/100g",
			"conversion_factor": 100,
		},
		"ocd": {
			"param": "ocd_mean",
			"description": "Organic carbon density",
			"unit": "kg/dm³",
			"conversion_factor": 10,
		},
		"ocs": {"param": "ocs_mean", "description": "Organic carbon stocks", "unit": "kg/m²", "conversion_factor": 10},
	}

	# Check params are valid
	assert set(params).issubset(param_dict), f"Invalid parameter. Must be one of {param_dict.keys()}"

	# Loop over parameters, load data and apply conversion factor to convert mapped data units to conventional units
	# adding description and unit to the image metadata
	images = [
		ee.Image(f"projects/soilgrids-isric/{param_dict[param]['param']}").divide(
			param_dict[param]["conversion_factor"]
		)
		for param in params
	]

	# Optionally, calculate the weighted mean of each parameter over the range 0 - `depth` [cm]
	if return_horizon_mean:
		# All images have the same horizons
		horizon_top = [0, 5, 15, 30, 60, 100]
		horizon_bottom = [5, 15, 30, 60, 100, 200]
		horizon_weights = tp.ee.stats._layer_weights(
			layer_tops=horizon_top, layer_bottoms=horizon_bottom, total_depth=depth
		)

		mean_images = []
		for i, img in enumerate(images):
			weight_dict = dict(zip(img.bandNames().getInfo(), horizon_weights, strict=False))
			mean_images.append(
				tp.ee.stats.weighted_mean(img=img, weight_dict=weight_dict, output_bandname=f"{params[i]}")
			)
		return ee.Image(mean_images)
	else:
		return ee.Image(images)


def usda_soil_class(sand: ee.Image = None, silt: ee.Image = None, clay: ee.Image = None) -> ee.Image:
	"""
	Takes images of sand/silt/clay percentage and returns USDA soil class (texture triangle)
	https://www.nrcs.usda.gov/sites/default/files/2022-09/The-Soil-Survey-Manual.pdf page 122

	Output Image has the same mask as `sand` image.

	1  "Sa": "sand"
	2  "LoSa": "loamy sand"
	3  "SaLo": "sandy loam"
	4  "Lo": "loam"
	5  "SiLo": "silty loam"
	6  "Si": "silt"
	7  "SaClLo": "sandy clay loam"
	8  "ClLo": "clay loam"
	9  "SiClLo": "silty clay loam"
	10  "SaCl": "sandy clay"
	11  "SiCl": "silty clay"
	12  "Cl": "clay"


	Args:
		sand: Image of sand percentage
		silt: Image of silt percentage
		clay: Image of clay percentage

	Returns:
		Image of soil class
	"""
	# Initialize soil class image with crs and crsTransform from `sand` image,
	# and set to be the same as that applied to `sand` image
	soil_class = (
		ee.Image(0)
		.setDefaultProjection(crs=sand.projection().wkt(), crsTransform=sand.getInfo()["bands"][0]["crs_transform"])
		.updateMask(sand)
	)

	# Sand - Material has more than 85 percent sand, and the percentage of silt plus
	# 1.5 times the percentage of clay is less than 15.
	soil_class = soil_class.where(sand.gt(85).And(silt.add(clay.multiply(1.5)).lt(15)), 1)
	# Loamy sands - Material has between 70 and 90 percent sand, the
	# percentage of silt plus 1.5 times the percentage of clay is 15 or more, and
	# the percentage of silt plus twice the percentage of clay is less than 30.
	soil_class = soil_class.where(
		sand.gte(70).And(sand.lte(90)).And(silt.add(clay.multiply(1.5)).gte(15)).And(silt.add(clay.multiply(2)).lt(30)),
		2,
	)
	# Sandy loams - Material has 7 to less than 20 percent clay and more
	# than 52 percent sand, and the percentage of silt plus twice the percentage
	# of clay is 30 or more; OR material has less than 7 percent clay and less
	# than 50 percent silt, and the percentage of silt plus twice the percentage
	# of clay is 30 or more.
	soil_class = soil_class.where(
		(clay.gte(7).And(clay.lt(20)).And(sand.gt(52)).And(silt.add(clay.multiply(2)).gte(30))).Or(
			clay.lt(7).And(silt.lt(50)).And(silt.add(clay.multiply(2)).gte(30))
		),
		3,
	)
	# Loam - Material has 7 to less than 27 percent clay, 28 to less than
	# 50 percent silt, and 52 percent or less sand.
	soil_class = soil_class.where(clay.gte(7).And(clay.lt(27)).And(silt.gte(28)).And(silt.lt(50)).And(sand.lte(52)), 4)
	# Silt loam - Material has 50 percent or more silt and 12 to less than
	# 27 percent clay; OR material has 50 to less than 80 percent silt and less
	# than 12 percent clay.
	soil_class = soil_class.where(
		(silt.gte(50).And(clay.gte(12)).And(clay.lt(27))).Or(silt.gte(50).And(silt.lt(80)).And(clay.lt(12))), 5
	)
	# Silt - Material has 80 percent or more silt and less than 12 percent
	# clay.
	soil_class = soil_class.where(silt.gte(80).And(clay.lt(12)), 6)
	# Sandy clay loam - Material has 20 to less than 35 percent clay, less
	# than 28 percent silt, and more than 45 percent sand.
	soil_class = soil_class.where(clay.gte(20).And(clay.lt(35)).And(silt.lt(28)).And(sand.gt(45)), 7)
	# Clay loam - Material has 27 to less than 40 percent clay and more
	# than 20 to 45 percent sand.
	soil_class = soil_class.where((clay.gte(27).And(clay.lt(40)).And(sand.gt(20)).And(sand.lte(45))), 8)
	# Silty clay loam - Material has 27 to less than 40 percent clay and 20
	# percent or less sand.
	soil_class = soil_class.where((clay.gte(27).And(clay.lt(40)).And(sand.lte(20))), 9)
	# Sandy clay - Material has 35 percent or more clay and more than
	# 45 percent sand.
	soil_class = soil_class.where(clay.gte(35).And(sand.gt(45)), 10)
	# Silty clay - Material has 40 percent or more clay and 40 percent or
	# more silt.
	soil_class = soil_class.where(clay.gte(40).And(silt.gte(40)), 11)
	# Clay - Material has 40 percent or more clay, 45 percent or less sand,
	# and less than 40 percent silt.
	soil_class = soil_class.where(clay.gte(40).And(sand.lte(45)).And(silt.lt(40)), 12)

	# Mask out null values
	soil_class = soil_class.selfMask()

	return soil_class.rename("texture")
