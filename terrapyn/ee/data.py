import typing as T

import ee

import terrapyn as tp


def soilgrids(
    params: T.List[str] = ["bdod", "cec", "cfvo", "clay", "sand", "silt", "nitrogen", "phh2o", "soc", "ocd", "ocs"],
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
        'clay', 'sand', 'silt', 'nitrogen', 'phh20', 'soc', 'ocd', 'ocs'.
        return_horizon_mean: If True, return the weighted mean over the range 0 - `depth` [cm]

    Returns:
        ee.Image with bands of the requested parameters
    """
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
        horizon_weights = tp.ee.stats._horizon_weights(
            horizon_top=horizon_top, horizon_bottom=horizon_bottom, topsoil_depth=depth
        )

        mean_images = []
        for i, img in enumerate(images):
            weight_dict = dict(zip(img.bandNames().getInfo(), horizon_weights))
            mean_images.append(
                tp.ee.stats.weighted_mean(img=img, weight_dict=weight_dict, output_bandname=f"{params[i]}")
            )
        return ee.Image(mean_images)
    else:
        return ee.Image(images)
