import datetime as dt
import typing as T

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
    Class to interact with the NOAA GSOD dataset in BigQuery.

    Available weather parameters are:
        - tavg: Average temperature (°C)
        - tmax: Maximum temperature (°C)
        - tmin: Minimum temperature (°C)
        - dewpoint: Dewpoint temperature (°C)
        - precip: Precipitation (mm)
    """

    def __init__(self):
        self.client = bigquery.Client()

    def __repr__(self):
        return "NOAA_GSOD()"

    def stations(
        self,
        start_date: dt.datetime = dt.date(1750, 1, 1),
        end_date: dt.datetime = dt.date.today(),
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
            where start_date > @start and end_date < @end
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
        station_ids: T.Union[str, T.List[str]],
        start_date: dt.datetime = dt.date(1750, 1, 1),
        end_date: dt.datetime = dt.date.today(),
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
        if isinstance(station_ids, str):
            station_ids = [station_ids]

        # TODO: Modify tables to search using years from date range
        # where regexp_contains(_TABLE_SUFFIX, '^[0-9]{4}$')
        query = """
            with raw as (
                select
                    concat(stn, wban) as id,
                    date,
                    case when `temp` = 9999.9 then null else (`temp` - 32.0) * (5.0/9.0) end as tavg,
                    case when `max` = 9999.9 then null else (`max` - 32.0) * (5.0/9.0) end as tmax,
                    case when `min` = 9999.9 then null else (`min` - 32.0) * (5.0/9.0) end as tmin,
                    case when dewp = 9999.9 then null else (dewp - 32.0) * (5.0/9.0) end as dewpoint,
                    case when prcp = 99.99 then null else prcp * 25.4 end as precip,
                from `bigquery-public-data.noaa_gsod.gsod*`
                where _TABLE_SUFFIX BETWEEN @start_year AND @end_year
                ),
                values as (
                    select * from raw
                    where tavg is not null or
                    tmax is not null or
                    tmin is not null or
                    dewpoint is not null or
                    precip is not null
                )
            select * from values
            where id in unnest(@station_ids)
            and date >= @start and date <= @end
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
