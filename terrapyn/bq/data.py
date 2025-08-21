import datetime as dt

import pandas as pd
import shapely
from google.cloud import bigquery

import terrapyn as tp
from terrapyn.logger import logger


def _shapely_geometry_to_string_for_bigquery(shapely_geometry: shapely.Geometry) -> str:
	"""
	Iterate through a list of tolerances and return the smallest geojson string that fits within the
	BigQuery limit of 1024k characters per SQL query. The max length is taken to be 1000k characters,
	reserving 24k characters for the rest of the query.
	"""
	tolerances = [
		0.000001,
		0.00001,
		0.0001,
		0.0005,
		0.001,
		0.005,
		0.01,
		0.02,
		0.03,
		0.04,
		0.05,
		0.06,
		0.07,
		0.08,
		0.09,
		0.1,
		0.2,
		0.3,
		0.4,
		0.5,
		0.6,
		0.7,
		0.8,
		0.9,
		1,
	]

	geometry_string = shapely.to_geojson(shapely_geometry)
	if tp.utils.utf8_len(geometry_string) < 1_000_000:
		return geometry_string
	else:
		for tolerance in tolerances:
			geometry = shapely.validation.make_valid(shapely_geometry.simplify(tolerance))
			geometry_string = shapely.to_geojson(geometry)
			if tp.utils.utf8_len(geometry_string) < 1_000_000:
				logger.warning(
					f"geojson string too long for BigQuery. Using simplified geometry with tolerance {tolerance}"
				)
				return geometry_string
		raise ValueError("No tolerance found that fits within the 1024k character limit")


class NOAA_GSOD:
	"""
	Class to interact with the NOAA Global Surface Summary of the Day (GSOD) dataset in BigQuery.
	"""

	def __init__(self):
		self.client = bigquery.Client()

	def __repr__(self):
		string = (
			"NOAA_GSOD: Class to interact with the NOAA GSOD dataset in BigQuery.\n"
			"Available weather parameters are:\n"
			" - tavg: Average temperature (°C)\n"
			" - tmax: Maximum temperature (°C)\n"
			" - tmin: Minimum temperature (°C)\n"
			" - dewpoint: Dewpoint temperature (°C)\n"
			" - precip: Precipitation (mm)\n"
			" - windspeed: Windspeed (m/s)"
		)
		return string

	def stations(
		self,
		start_date: dt.date | None = None,
		end_date: dt.date | None = None,
		geom: shapely.Geometry = None,
	):
		"""
		Get weather stations from the NOAA GSOD dataset that are within a given geometry and date range.

		Args:
			start_date: The start date for the station data.
			end_date: The end date for the station data.
			geom: A shapely geometry object to filter by.

		Returns:
			A Pandas DataFrame with the station metadata.
		"""
		if geom is None:
			geom = shapely.geometry.box(-180, -90, 180, 90)

		if start_date is None:
			start_date = dt.date(1750, 1, 1)
		elif isinstance(start_date, dt.datetime):
			start_date = start_date.date()

		if end_date is None:
			end_date = dt.date.today()
		elif isinstance(end_date, dt.datetime):
			end_date = end_date.date()

		geometry_string = _shapely_geometry_to_string_for_bigquery(geom)

		query = """
			with stations as (
				select
					concat(usaf, wban) as id,
					st_geogpoint(lon, lat) as geom,
					parse_numeric(elev) as elevation,
					parse_date('%Y%m%d', `begin`) as start_date,
					parse_date('%Y%m%d', `end`) as end_date,
				from `bigquery-public-data.noaa_gsod.stations`
				where lat != 0 and lon != 0
			)
			select * from stations
			where start_date <= @end and end_date >= @start
			and st_intersects(geom, st_geogfromgeojson(@geometry_string, make_valid => True))
			"""

		job_config = bigquery.QueryJobConfig(
			query_parameters=[
				bigquery.ScalarQueryParameter("start", "DATE", start_date),
				bigquery.ScalarQueryParameter("end", "DATE", end_date),
				bigquery.ScalarQueryParameter("geometry_string", "STRING", geometry_string),
			]
		)
		query_job = self.client.query(query, job_config=job_config)
		df = query_job.to_geodataframe()

		df["start_date"] = pd.to_datetime(df["start_date"])
		df["end_date"] = pd.to_datetime(df["end_date"])
		df.loc[df["elevation"].isna() | df["elevation"].eq(-999), "elevation"] = 0.0
		df["elevation"] = df["elevation"].astype(float)
		return df

	def data(
		self,
		station_ids: str | list[str],
		start_date: dt.date | None = None,
		end_date: dt.date | None = None,
	):
		"""
		Get weather data from the NOAA GSOD dataset for the given stations and date range.

		Args:
			station_ids: The station ID(s).
			start_date: The start date for the station data.
			end_date: The end date for the station data.

		Returns:
			A Pandas DataFrame with the station data.
		"""
		station_ids = tp.utils.ensure_list(station_ids)

		if start_date is None:
			start_date = dt.date(1750, 1, 1)
		elif isinstance(start_date, dt.datetime):
			start_date = start_date.date()

		if end_date is None:
			end_date = dt.date.today()
		elif isinstance(end_date, dt.datetime):
			end_date = end_date.date()

		if isinstance(end_date, dt.datetime):
			end_date = end_date.date()

		query = """
			with raw as (
				select
					concat(stn, wban) as id,
					parse_date('%Y%m%d', concat(year, mo, da)) date,
					if(`temp` = 9999.9, null, (`temp` - 32.0) * (5.0/9.0)) as tavg,
					if(`max` = 9999.9, null, (`max` - 32.0) * (5.0/9.0)) as tmax,
					if(`min` = 9999.9, null, (`min` - 32.0) * (5.0/9.0)) as tmin,
					if(dewp = 9999.9, null, (dewp - 32.0) * (5.0/9.0)) as dewpoint,
					if(prcp = 99.99, null, prcp * 25.4) as precip,
					if(wdsp = '999.9', null, cast(wdsp as float64) * 0.5144444444) as windspeed,
				from `bigquery-public-data.noaa_gsod.gsod*`
				where _TABLE_SUFFIX BETWEEN @start_year AND @end_year
				)
			select * from raw
			where id in unnest(@station_ids) and date >= @start and date <= @end and (
				tavg is not null or
				tmax is not null or
				tmin is not null or
				dewpoint is not null or
				precip is not null or
				windspeed is not null
			)
			"""

		job_config = bigquery.QueryJobConfig(
			query_parameters=[
				bigquery.ArrayQueryParameter("station_ids", "STRING", station_ids),
				bigquery.ScalarQueryParameter("start", "DATE", start_date),
				bigquery.ScalarQueryParameter("end", "DATE", end_date),
				bigquery.ScalarQueryParameter("start_year", "STRING", str(start_date.year)),
				bigquery.ScalarQueryParameter("end_year", "STRING", str(end_date.year)),
			]
		)
		query_job = self.client.query(query, job_config=job_config)
		df = query_job.to_dataframe()
		df["date"] = pd.to_datetime(df["date"])
		df = df.sort_values(by=["id", "date"]).reset_index(drop=True)
		return df


class NOAA_GHCN:
	"""
	Class to interact with the NOAA Global Historical Climatology Network daily (GHCNd) dataset in BigQuery.
	"""

	def __init__(self):
		self.client = bigquery.Client()

	def __repr__(self):
		string = (
			"NOAA_GHCN: Class to interact with the NOAA GHCN dataset in BigQuery.\n"
			"Available weather parameters are:\n"
			" - tmax: Maximum temperature (°C)\n"
			" - tmin: Minimum temperature (°C)\n"
			" - precip: Precipitation (mm)\n"
			" - snow: Snowfall (mm)\n"
		)
		return string

	def stations(
		self,
		start_date: dt.date | None = None,
		end_date: dt.date | None = None,
		geom: shapely.Geometry = None,
	):
		"""
		Get weather stations from the NOAA GHCN dataset that are within a given geometry and date range.

		Args:
			start_date: The start date for the station data.
			end_date: The end date for the station data.
			geom: A shapely geometry object to filter by.

		Returns:
			A Pandas DataFrame with the station metadata.
		"""
		if geom is None:
			geom = shapely.geometry.box(-180, -90, 180, 90)

		if start_date is None:
			start_date = dt.date(1750, 1, 1)
		elif isinstance(start_date, dt.datetime):
			start_date = start_date.date()

		if end_date is None:
			end_date = dt.date.today()
		elif isinstance(end_date, dt.datetime):
			end_date = end_date.date()

		geometry_string = _shapely_geometry_to_string_for_bigquery(geom)

		query = """
			with stations as (
				select
					a.id,
					st_geogpoint(a.longitude, a.latitude) as geom,
					a.elevation,
					date(b.start_year, 1, 1) as start_date,
					date(b.end_year, 12, 31) as end_date,
				from `bigquery-public-data.ghcn_d.ghcnd_stations` a
				join (
					select
						id,
						min(firstyear) as start_year,
						max(lastyear) as end_year
					from `bigquery-public-data.ghcn_d.ghcnd_inventory`
					group by id
					) b
				using(id)
			)
			select * from stations
			where start_date <= @end and end_date >= @start
			and st_intersects(geom, st_geogfromgeojson(@geometry_string, make_valid => True))
			"""

		job_config = bigquery.QueryJobConfig(
			query_parameters=[
				bigquery.ScalarQueryParameter("start", "DATE", start_date),
				bigquery.ScalarQueryParameter("end", "DATE", end_date),
				bigquery.ScalarQueryParameter("geometry_string", "STRING", geometry_string),
			]
		)
		query_job = self.client.query(query, job_config=job_config)
		df = query_job.to_geodataframe()

		df["start_date"] = pd.to_datetime(df["start_date"])
		df["end_date"] = pd.to_datetime(df["end_date"])
		df.loc[df["elevation"].isna() | df["elevation"].eq(-999.9), "elevation"] = 0.0
		df["elevation"] = df["elevation"].astype(float)
		return df

	def data(
		self,
		station_ids: str | list[str],
		start_date: dt.date | None = None,
		end_date: dt.date | None = None,
	):
		"""
		Get weather data from the NOAA GHCN dataset for the given stations and date range.

		Args:
			station_ids: The station ID(s).
			start_date: The start date for the station data.
			end_date: The end date for the station data.

		Returns:
			A Pandas DataFrame with the station data.
		"""
		station_ids = tp.utils.ensure_list(station_ids)

		if start_date is None:
			start_date = dt.date(1750, 1, 1)
		elif isinstance(start_date, dt.datetime):
			start_date = start_date.date()

		if end_date is None:
			end_date = dt.date.today()
		elif isinstance(end_date, dt.datetime):
			end_date = end_date.date()

		query = """
			with raw as (
				select
					id,
					`date`,
					if(element = 'TMAX', value/10, null) as tmax,
					if(element = 'TMIN', value/10, null) as tmin,
					if(element = 'PRCP', value/10, null) as precip,
					if(element = 'SNOW', value, null) as snow
				from `bigquery-public-data.ghcn_d.ghcnd_*`
				where _TABLE_SUFFIX BETWEEN @start_year AND @end_year
				and qflag is null
				)
			select
				id,
				`date`,
				max(tmax) as tmax,
				max(tmin) as tmin,
				max(precip) as precip,
				max(snow) as snow
			from raw
			where id in unnest(@station_ids) and `date` >= @start and `date` <= @end and (
				tmax is not null or
				tmin is not null or
				precip is not null or
				snow is not null
			)
			group by id, `date`
			order by id, `date`
			"""

		job_config = bigquery.QueryJobConfig(
			query_parameters=[
				bigquery.ArrayQueryParameter("station_ids", "STRING", station_ids),
				bigquery.ScalarQueryParameter("start", "DATE", start_date),
				bigquery.ScalarQueryParameter("end", "DATE", end_date),
				bigquery.ScalarQueryParameter("start_year", "STRING", str(start_date.year)),
				bigquery.ScalarQueryParameter("end_year", "STRING", str(end_date.year)),
			]
		)
		query_job = self.client.query(query, job_config=job_config)
		df = query_job.to_dataframe()
		df["date"] = pd.to_datetime(df["date"])
		return df
