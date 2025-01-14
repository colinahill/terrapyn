import numpy as np


def growing_degree_days(
	tmax: np.ndarray = None, tmin: np.ndarray = None, t_base: float = 0, t_cutoff: float = None
) -> np.ndarray:
	"""
	Calculate the modified growing (thermal) degree days (GDD/TDD) from daily maximum and minimum temperatures.
	If `t_cutoff` is given, `tmax` is thresholded to `t_cutoff`. If the daily mean of the adjusted tmax and
	tmin is below `t_base`, the daily value is set to 0.

	Args:
		tmax: Daily maximum temperature.
		tmin: Daily minimum temperature.
		t_base: Base temperature, below which the rate of growth or development is zero.
		t_cutoff: Upper developmental threshold/cutoff temperature, above which the rate of growth or
		development begins to decrease or stop.
	"""
	if tmax is None or tmin is None:
		raise ValueError("Both tmax and tmin must be provided.")

	if t_cutoff is None:
		t_cutoff = np.inf

	gdd = np.clip((np.clip(tmax, a_min=None, a_max=t_cutoff) + tmin) / 2 - t_base, a_min=0, a_max=None)
	return gdd
