import typing as T

import numpy as np
import pandas as pd

import terrapyn as tp

from .metrics import (
	bias_df,
	efficiency_df,
	error_df,
	mae_df,
	me_df,
	mse_df,
	normalized_mae_df,
	normalized_mse_df,
	normalized_rmse_df,
	r2,
	rmse_df,
)

DATAFRAME_METRICS = {
	"me": me_df,
	"mae": mae_df,
	"rmse": rmse_df,
	"mse": mse_df,
	"error": error_df,
	"bias": bias_df,
	"efficiency": efficiency_df,
	"nmae": normalized_mae_df,
	"nmse": normalized_mse_df,
	"nrmse": normalized_rmse_df,
	"r2": r2,
}


def score_df(
	df: pd.DataFrame = None,
	metric: str = "mae",
	model_names: str | list[str] = None,
	obs_names: str | list[str] = None,
	output_index_names: T.Iterable = None,
	axis: int = 0,
):
	"""
	Score columns of values in a Pandas DataFrame. Can compare single/multiple columns at once (in order), so multiple
	parameters can be scored simultaneously.

	If multiple values of `model_names` and/or `obs_names` are given, the scores are computed as follows:
		- Equal lengths of `model_names` and `obs_names` results in the pair-wise scores (column 1 with 1,
		column 2 with 2 etc.).
		- If `model_names` has more elements than `obs_names`: Scores are computed as all 'model' columns with
		the first 'obs' column, then all 'model' columns with the second 'obs' column etc.

	Args:
		df: DataFrame with columns of values to compare.
		metric: Name of scoring metric, one of "me", mae", "rmse", "mse", "error", "bias", "efficiency"
		model_names: Model name(s) that will be used to compare to the observations `obs_names`.
		obs_names: Observation name(s) that will be used to compare to the observations `obs_names`.
		output_index_names: The names given to the output columns. Unique names are automatically generated
		by default in the form <model_name>_<obs_name>
		axis: The axis over which to compute the scores (column-wise=0)
	"""
	if metric not in DATAFRAME_METRICS:
		options = ", ".join(DATAFRAME_METRICS.keys())
		raise ValueError(f"`metric` must be one of {options}")

	return DATAFRAME_METRICS[metric](
		df=df, model_name=model_names, obs_name=obs_names, output_index_names=output_index_names, axis=axis
	)


def _set_to_nan_if_fewer_than_min_points(
	x: pd.DataFrame | pd.Series, min_points: int = None
) -> pd.DataFrame | pd.Series:
	reject_mask = np.isfinite(x).sum() < min_points
	x.loc[:, reject_mask] = np.nan
	return x


def grouped_scores(
	df: pd.DataFrame = None,
	metrics: str | list = "mae",
	groupby_time: bool = True,
	time_dim: str = "time",
	time_grouping: str = "month",
	other_grouping_keys: str | list[str] = None,
	min_points: int = None,
	model_names: str | list[str] = None,
	obs_names: str | list[str] = None,
	output_index_names: str | list[str] = None,
) -> pd.DataFrame:
	"""
	Scores values in a Pandas DataFrame grouped by keys. Can compare single/multiple columns at once (in order),
	so multiple parameters can be scored simultaneously. Can score using multiple metrics, and multiple grouping keys.

	If multiple values of `model_names` and/or `obs_names` are given, the scores are computed as follows:
		- Equal lengths of `model_names` and `obs_names` results in the pair-wise scores (column 1 with 1,
		column 2 with 2 etc.).
		- If `model_names` has more elements than `obs_names`: Scores are computed as all 'model' columns with
		the first 'obs' column, then all 'model' columns with the second 'obs' column etc.

	Args:
		df: DataFrame with columns of values to compare
		metrics: Names of scoring metric, choose from "me", mae", "rmse", "mse", "error", "bias", "efficiency".
		groupby_time: If True, group by time, otherwise group by other grouping keys.
		time_dim: The name of the time dimension in the DataFrame.
		time_grouping: The time grouping to use if `grouping_keys` includes `'time'`, one of 'week',
		'month', 'dayofyear', 'dekad', 'pentad', 'year'.
		other_grouping_keys: Keys used to group the data, not including `time_dim` (can be index or columns names).
		min_points: Minimum number of points required to calculate the score, otherwise return `NaN`.
		model_names: Model name(s) that will be used to compare to the observations `obs_names`.
		obs_names: Observation name(s) that will be used to compare to the observations `obs_names`.
		output_index_names: The names given to the ourput columns/index. Unique names are automatically generated
		by default in the form <model_name>_<obs_name>.
		axis: The axis over which to compute the scores (column-wise=0).

	Returns:
		Pandas DataFrame with scores for each group, for each metric, for each combination of model and observation
	"""
	if other_grouping_keys is None:
		# If no other grouping keys are given, only group by time
		if groupby_time is False:
			# TODO apply metrics to the whole DataFrame
			pass
	elif isinstance(other_grouping_keys, str):
		other_grouping_keys = [other_grouping_keys]

	# Check if grouping by time
	if groupby_time:
		grouped = tp.time.groupby_time(
			df, time_dim=time_dim, grouping=time_grouping, other_grouping_keys=other_grouping_keys
		)
	else:
		grouped = df.groupby(other_grouping_keys)

	# If `min_points` is given, create groups and set column values in each group to NaN if fewer than `min_points`.
	# Then recreate grouped data
	if min_points is not None:
		df_min_points = grouped.apply(_set_to_nan_if_fewer_than_min_points, min_points=min_points)

		if groupby_time:
			grouped = tp.time.groupby_time(
				df_min_points, time_dim=time_dim, grouping=time_grouping, other_grouping_keys=other_grouping_keys
			)
		else:
			grouped = df_min_points.groupby(other_grouping_keys)

	if isinstance(metrics, str):
		scores = grouped.apply(
			score_df,
			metric=metrics,
			model_names=model_names,
			obs_names=obs_names,
			output_index_names=output_index_names,
			axis=0,
			include_groups=False,
		)
	else:
		df_list = []
		for metric in metrics:
			scores = grouped.apply(
				score_df,
				metric=metric,
				model_names=model_names,
				obs_names=obs_names,
				output_index_names=output_index_names,
				axis=0,
				include_groups=False,
			)

			scores.columns = pd.MultiIndex.from_product(
				[output_index_names if output_index_names is not None else scores.columns, [metric]]
			)

			if groupby_time:
				# `scores` has the `time_grouping` dimension as the as the first index (level=0)
				scores.index = scores.index.rename(time_grouping, level=0)

			df_list.append(scores)

		scores = pd.concat(df_list, axis=1)
		scores = scores.sort_index(axis=1)

	scores = scores.sort_index(axis=0)

	return scores
