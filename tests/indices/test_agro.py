import numpy as np

from terrapyn.indices import growing_degree_days


def test_growing_degree_days():
	tmax = np.array([10, 20, 30, 40])
	tmin = np.array([-10, 0, 10, 20])
	t_base = 10
	t_cutoff = 30
	expected = np.array([0.0, 0.0, 10.0, 15.0])
	result = growing_degree_days(tmax, tmin, t_base, t_cutoff)
	np.testing.assert_array_equal(result, expected)
	expected = np.array([0.0, 0.0, 10.0, 20.0])
	result = growing_degree_days(tmax, tmin, t_base, t_cutoff=None)
	np.testing.assert_array_equal(result, expected)
