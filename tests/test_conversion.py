import unittest

import numpy as np
import pandas as pd
import xarray as xr

import terrapyn as tp

SSRD = 13.28  # MJ m-2 day-1
WS = 2.078  # m s-1
T = 16.9  # degC
SVP = 1.997  # kPa
AVP = 1.409  # kPa
DELTA_SVP = 0.122  # kPa degC-1
PSY = 0.0666  # kPa degC-1
ELEVATION = 100  # m
ATMOS_PRES = 100.1  # kPa
TMAX = 21.5  # degC
TMIN = 12.3  # degC
WIND10M = 2.778  # m s-1
RHMAX = 84  # %
RHMIN = 63  # %
RHMEAN = 73.5  # %
SVP_TMIN = 1.431  # kPa
SVP_TMAX = 2.564  # kPa
LATITUDE = 50.8  # degrees
TWET = 19.5  # degC
TDRY = 25.6  # degC
TDEW = 14.8  # degC

n_lat = 3
n_lon = 3
n_time = 3
n_total = n_lat * n_lon * n_time
lats = pd.Index(np.linspace(43, 45, n_lat), name="lat")
lons = pd.Index(np.linspace(1, 3, n_lon), name="lon")
times = pd.date_range("20210401", freq="D", periods=n_time, name="time")
index = pd.MultiIndex.from_product([lats, lons, times])
df = pd.DataFrame(
	{
		"elevation": [ELEVATION] * n_total,  # m
		"tmax": [TMAX] * n_total,  # ˚C
		"tmin": [TMIN] * n_total,  # ˚C
		"rhmax": [RHMAX] * n_total,  # %
		"rhmin": [RHMIN] * n_total,  # %
		"wind10m": [WIND10M] * n_total,  # m/s
		"ssrd": [SSRD] * n_total,  # MJ m-2 day-1
		"ws": [WS] * n_total,  # m/s
		"t": [T] * n_total,  # degC
		"svp": [SVP] * n_total,  # kPa
		"avp": [AVP] * n_total,  # kPa
		"delta_svp": [DELTA_SVP] * n_total,  # kPa degC-1
		"psy": [PSY] * n_total,  # kPa degC-1
	},
	index=index,
)


class TestWindSpeed2m(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.wind_speed_2m(3.2, 10)
		np.testing.assert_almost_equal(result, 2.4, decimal=1)

	def test_series(self):
		result = tp.conversion.wind_speed_2m(df["wind10m"], 10).loc[44.0, 2.0, "2021-04-02"]
		np.testing.assert_almost_equal(result, 2.0778080868165487)


class TestSvpFromT(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.saturation_vapour_pressure_from_temperature(T)
		np.testing.assert_almost_equal(result, 1.9215514597960852)

	def test_series(self):
		result = tp.conversion.saturation_vapour_pressure_from_temperature(df["t"]).loc[44.0, 2.0, "2021-04-02"]
		np.testing.assert_almost_equal(result, 1.9215514597960852)


class TestDeltaSvp(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.delta_saturation_vapour_pressure(T)
		np.testing.assert_almost_equal(result, 0.12186328504483228)

	def test_series(self):
		result = tp.conversion.delta_saturation_vapour_pressure(df["t"]).loc[44.0, 2.0, "2021-04-02"]
		np.testing.assert_almost_equal(result, 0.12186328504483228)


class TestAtmPressure(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.atmospheric_pressure(ELEVATION)
		np.testing.assert_almost_equal(result, 100.12350828341812)

	def test_series(self):
		result = tp.conversion.atmospheric_pressure(df["elevation"]).loc[44.0, 2.0, "2021-04-02"]
		np.testing.assert_almost_equal(result, 100.12350828341812)


class TestPsyConstant(unittest.TestCase):
	def test_altitude_single_value(self):
		result = tp.conversion.psychrometric_constant(altitude=ELEVATION)
		np.testing.assert_almost_equal(result, 0.06658213300847304)

	def test_atmos_pres_single_value(self):
		result = tp.conversion.psychrometric_constant(atmos_pres=ATMOS_PRES)
		np.testing.assert_almost_equal(result, 0.0665665)

	def test_series_altitude(self):
		result = tp.conversion.psychrometric_constant(altitude=df["elevation"]).loc[44.0, 2.0, "2021-04-02"]
		np.testing.assert_almost_equal(result, 0.06658213300847304)


class TestAvpFromRhminRhmax(unittest.TestCase):
	def test_temp_and_rh_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_rhmin_rhmax(tmin=TMIN, tmax=TMAX, rh_min=RHMIN, rh_max=RHMAX)
		np.testing.assert_almost_equal(result, 1.4058885969470276)

	def test_svp_and_rh_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_rhmin_rhmax(
			svp_tmin=SVP_TMIN, svp_tmax=SVP_TMAX, rh_min=RHMIN, rh_max=RHMAX
		)
		np.testing.assert_almost_equal(result, 1.40868)


class TestAvpFromRhmax(unittest.TestCase):
	def test_svptmin_and_rhmax_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_rhmax(SVP_TMIN, RHMAX)
		np.testing.assert_almost_equal(result, 1.20204)

	def test_rhmax_and_tmin_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_rhmax(rh_max=RHMAX, tmin=TMIN)
		np.testing.assert_almost_equal(result, 1.1995003886672477)


class TestAvpFromTdew(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_tdew(T)
		np.testing.assert_almost_equal(result, 1.9215514597960852)


class TestAvpFromTmin(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_tmin(TMIN)
		np.testing.assert_almost_equal(result, 1.4279766531752949)


class TestAvpFromRhmean(unittest.TestCase):
	def test_svp_and_rh_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_rhmean(SVP_TMIN, SVP_TMAX, RHMEAN)
		np.testing.assert_almost_equal(result, 1.4681625)

	def test_rh_and_temp_single_value(self):
		result = tp.conversion.actual_vapour_pressure_from_rhmean(None, None, RHMEAN, TMIN, TMAX)
		np.testing.assert_almost_equal(result, 1.4652762230908918)


class TestDeg2Rad(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.degrees_to_radians(LATITUDE)
		np.testing.assert_almost_equal(result, 0.8866272600131193)


class TestRad2Deg(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.radians_to_degrees(0.3)
		np.testing.assert_almost_equal(result, 17.188733853924695)


class TestCelsius2Kelvin(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.celsius_to_kelvin(20)
		self.assertEqual(result, 293.15)


class TestKelvin2Celsius(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.kelvin_to_celsius(293.15)
		self.assertEqual(result, 20)


class TestKilometersPerHour2MetersPerSecond(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.kilometers_per_hour_to_meters_per_second(37)
		np.testing.assert_almost_equal(result, 10.277777777777777)


class TestAvpFromTwetTdry(unittest.TestCase):
	def test_single_value(self):
		atmos_pres = tp.conversion.atmospheric_pressure(1200)
		psy_const = tp.conversion.psychrometric_constant_of_psychrometer(1, atmos_pres)
		svp_twet = tp.conversion.saturation_vapour_pressure_from_temperature(TWET)
		result = tp.conversion.actual_vapour_pressure_from_twet_tdry(TWET, TDRY, svp_twet, psy_const)
		np.testing.assert_almost_equal(result, 1.9072375322669735)


class TestPsyConstOfPsychronometer(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.psychrometric_constant_of_psychrometer(1, ATMOS_PRES)
		np.testing.assert_almost_equal(result, 0.0662662)
		result = tp.conversion.psychrometric_constant_of_psychrometer(2, ATMOS_PRES)
		np.testing.assert_almost_equal(result, 0.08008)
		result = tp.conversion.psychrometric_constant_of_psychrometer(3, ATMOS_PRES)
		np.testing.assert_almost_equal(result, 0.12011999999999998)


class TestActualVapourPressure(unittest.TestCase):
	def test_first_method_single_value(self):
		result = tp.conversion.actual_vapour_pressure(tdew=TDEW)
		np.testing.assert_almost_equal(result, 1.6802076416421783)

	def test_second_method_single_value(self):
		result = tp.conversion.actual_vapour_pressure(twet=TWET, tdry=TDRY, atmos_pres=ATMOS_PRES)
		np.testing.assert_almost_equal(result, 1.8561260693675472)

	def test_third_method_single_value(self):
		result = tp.conversion.actual_vapour_pressure(tmin=TMIN, tmax=TMAX, rh_min=RHMIN, rh_max=RHMAX)
		np.testing.assert_almost_equal(result, 1.4058885969470276)

	def test_fourth_method_single_value(self):
		result = tp.conversion.actual_vapour_pressure(tmin=TMIN, rh_max=RHMAX)
		np.testing.assert_almost_equal(result, 1.1995003886672477)

	def test_fifth_method_single_value(self):
		result = tp.conversion.actual_vapour_pressure(rh_mean=RHMEAN, tmin=TMIN, tmax=TMAX)
		np.testing.assert_almost_equal(result, 1.4652762230908918)

	def test_sixth_method_single_value(self):
		result = tp.conversion.actual_vapour_pressure(tmin=TMIN)
		np.testing.assert_almost_equal(result, 1.4279766531752949)

	def test_raise_value_error(self):
		with self.assertRaises(ValueError):
			tp.conversion.actual_vapour_pressure(tmax=TMAX)


class TestMonthlySoilHeatFlux(unittest.TestCase):
	tmean_month_1 = 14.1
	tmean_month_2 = 16.1
	tmean_month_3 = 18.8

	def test_next_month_single_value(self):
		result = tp.conversion.monthly_soil_heat_flux(self.tmean_month_1, self.tmean_month_3, next_month=True)
		np.testing.assert_almost_equal(result, 0.33, decimal=1)

	def test_current_month_single_value(self):
		result = tp.conversion.monthly_soil_heat_flux(self.tmean_month_1, self.tmean_month_2)
		np.testing.assert_almost_equal(result, 0.28, decimal=1)


class testVectorsToScalar(unittest.TestCase):
	time = pd.date_range("2021-01-01", "2021-01-31")
	lat = range(0, 6)
	lon = range(0, 6)
	u10 = np.ones((len(lat), len(lon), time.size)) * np.arange(time.size)
	v10 = np.ones((len(lat), len(lon), time.size)) * np.arange(time.size)
	ds = xr.Dataset(
		data_vars=dict(
			u10=(["lat", "lon", "time"], u10),
			v10=(["lat", "lon", "time"], v10),
		),
		coords=dict(lat=lat, lon=lon, time=time),
	)

	def test_value(self):
		exp = np.ones((len(self.lat), len(self.lon), self.time.size)) * np.sqrt(
			2 * np.arange(self.time.size) * np.arange(self.time.size)
		)

		obs = tp.conversion.vectors_to_scalar(self.ds["u10"], self.ds["v10"])
		np.testing.assert_almost_equal(exp, obs)


class testWindDirection(unittest.TestCase):
	time = pd.date_range("2021-01-01", "2021-01-16")
	lat = range(0, 6)
	lon = range(0, 6)
	tan30 = 1 / np.sqrt(3)
	tan60 = np.sqrt(3)
	ones = np.ones((len(lat), len(lon), time.size))
	u10 = ones * np.array([1, 0, -1, 0, 1, 1, -1, -1, 1, 1, tan60, tan30, -1, -1, -tan60, -tan30])
	v10 = ones * np.array([0, 1, 0, -1, 1, -1, -1, 1, -tan60, -tan30, 1, 1, tan60, tan30, -1, -1])
	ds = xr.Dataset(
		data_vars=dict(
			u10=(["lat", "lon", "time"], u10),
			v10=(["lat", "lon", "time"], v10),
		),
		coords=dict(lat=lat, lon=lon, time=time),
	)

	def test_values(self):
		expected = self.ones * np.array(
			[270.0, 180.0, 90.0, 0.0, 225.0, 315.0, 45.0, 135.0, 330.0, 300.0, 240.0, 210.0, 150.0, 120.0, 60.0, 30.0]
		)

		result = tp.conversion.wind_direction(u=self.ds["u10"], v=self.ds["v10"], convention="from", unit="deg")
		np.testing.assert_almost_equal(expected, result)

	def test_unit_rads(self):
		exp = self.ones * np.array(
			[270.0, 180.0, 90.0, 0.0, 225.0, 315.0, 45.0, 135.0, 330.0, 300.0, 240.0, 210.0, 150.0, 120.0, 60.0, 30.0]
		)
		exp = exp * np.pi / 180

		obs = tp.conversion.wind_direction(u=self.ds["u10"], v=self.ds["v10"], convention="from", unit="rad")
		np.testing.assert_almost_equal(exp, obs)

	def test_direction_to(self):
		exp = self.ones * np.array(
			[270.0, 180.0, 90.0, 0.0, 225.0, 315.0, 45.0, 135.0, 330.0, 300.0, 240.0, 210.0, 150.0, 120.0, 60.0, 30.0]
		)
		exp = exp - 180
		exp = np.where(exp >= 0, exp, exp + 360)

		obs = tp.conversion.wind_direction(u=self.ds["u10"], v=self.ds["v10"], convention="to", unit="deg")
		np.testing.assert_almost_equal(exp, obs)

	def test_zero_wind(self):
		obs = tp.conversion.wind_direction(u=0, v=0)
		self.assertAlmostEqual(0, obs)

	def test_invalid_unit(self):
		with self.assertRaises(ValueError):
			tp.conversion.wind_direction(u=self.ds["u10"], v=self.ds["v10"], convention="from", unit="apple")

	def test_invalid_convention(self):
		with self.assertRaises(ValueError):
			tp.conversion.wind_direction(u=self.ds["u10"], v=self.ds["v10"], convention="oranges", unit="deg")


class TestRelativeHumidityFromActualAndSaturatedVapourPressure(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.relative_humidity_from_actual_vapour_pressure_and_saturated_vapour_pressure(AVP, SVP)
		np.testing.assert_almost_equal(result, 70.55583375062594)


class TestRelativeHumidityFromDewPointAndTemperature(unittest.TestCase):
	def test_single_value(self):
		result = tp.conversion.relative_humidity_from_dew_point_and_temperature(TDEW, T)
		np.testing.assert_almost_equal(result, 87.44015847592662)
