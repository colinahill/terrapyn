# USDA soil textures: https://www.nrcs.usda.gov/sites/default/files/2022-09/The-Soil-Survey-Manual.pdf page 122
USDA_SOIL_CLASS = {
	1: "sand",
	2: "loamy sand",
	3: "sandy loam",
	4: "loam",
	5: "silt loam",
	6: "silt",
	7: "sandy clay loam",
	8: "clay loam",
	9: "silty clay loam",
	10: "sandy clay",
	11: "silty clay",
	12: "clay",
}


def usda_soil_class(sand=None, silt=None, clay=None):
	"""
	Takes floats of sand/silt/clay percentage and returns USDA soil class (texture triangle)
	https://www.nrcs.usda.gov/sites/default/files/2022-09/The-Soil-Survey-Manual.pdf page 122

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

	Returns:
		integer soil class
	"""
	# Sand - Material has more than 85 percent sand, and the percentage of silt plus
	# 1.5 times the percentage of clay is less than 15.
	if (sand > 85) and (silt + 1.5 * clay < 15):
		soil_class = 1
	# Loamy sands - Material has between 70 and 90 percent sand, the
	# percentage of silt plus 1.5 times the percentage of clay is 15 or more, and
	# the percentage of silt plus twice the percentage of clay is less than 30.
	elif (sand >= 70) and (sand <= 90) and (silt + 1.5 * clay >= 15) and (silt + 2 * clay < 30):
		soil_class = 2
	# Sandy loams - Material has 7 to less than 20 percent clay and more
	# than 52 percent sand, and the percentage of silt plus twice the percentage
	# of clay is 30 or more; OR material has less than 7 percent clay and less
	# than 50 percent silt, and the percentage of silt plus twice the percentage
	# of clay is 30 or more.
	elif ((clay >= 7) and (clay < 20) and (sand > 52) and (silt + 2 * clay >= 30)) or (
		(clay < 7) and (silt < 50) and (silt + 2 * clay >= 30)
	):
		soil_class = 3
	# Loam - Material has 7 to less than 27 percent clay, 28 to less than
	# 50 percent silt, and 52 percent or less sand.
	elif (clay >= 7) and (clay < 27) and (silt >= 28) and (silt < 50) and (sand <= 52):
		soil_class = 4
	# Silt loam - Material has 50 percent or more silt and 12 to less than
	# 27 percent clay; OR material has 50 to less than 80 percent silt and less
	# than 12 percent clay.
	elif ((silt >= 50) and (clay >= 12) and (clay < 27)) or ((silt >= 50) and (silt < 80) and (clay < 12)):
		soil_class = 5
	# Silt - Material has 80 percent or more silt and less than 12 percent
	# clay.
	elif (silt >= 80) and (clay < 12):
		soil_class = 6
	# Sandy clay loam - Material has 20 to less than 35 percent clay, less
	# than 28 percent silt, and more than 45 percent sand.
	elif (clay >= 20) and (clay < 35) and (silt < 28) and (sand > 45):
		soil_class = 7
	# Clay loam - Material has 27 to less than 40 percent clay and more
	# than 20 to 45 percent sand.
	elif (clay >= 27) and (clay < 40) and (sand > 20) and (sand <= 45):
		soil_class = 8
	# Silty clay loam - Material has 27 to less than 40 percent clay and 20
	# percent or less sand.
	elif (clay >= 27) and (clay < 40) and (sand <= 20):
		soil_class = 9
	# Sandy clay - Material has 35 percent or more clay and more than
	# 45 percent sand.
	elif (clay >= 35) and (sand > 45):
		soil_class = 10
	# Silty clay - Material has 40 percent or more clay and 40 percent or
	# more silt.
	elif (clay >= 40) and (silt >= 40):
		soil_class = 11
	# Clay - Material has 40 percent or more clay, 45 percent or less sand,
	# and less than 40 percent silt.
	elif (clay >= 40) and (sand <= 45) and (silt < 40):
		soil_class = 12
	return soil_class
