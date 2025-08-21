import typing as T

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

import terrapyn as tp
from terrapyn.logger import logger


class ConfusionMatrix:
	"""
	Calculate a confusion matrix (contingency table) for observations (truth) and a model (predictions),
	with associated statistics.

	Args:
		truth: Array of values for observations (truth).
		prediction: Array of values for a model (predictions).
		normalize: Normalize the confusion matrix over the truth (rows), prediction (columns) conditions,
		or all the population, one of 'true', 'pred', 'all', None. If None, confusion matrix will not be normalized.
		labels_to_include: List of labels in the truth/prediction arrays to include when calculating the confusion
		matrix, such that all other labels will be ignored. If `None` include all labels.

	Confusion Matrix Statistics. Available attributes are:

	==========    ================================================================
	ACC	   Accuracy
	F1score       F1 score
	FDR	   False Discovery Rate
	FNR	   False Negative Rate (Miss rate)
	FOR	   False Omission Rate
	FPR	   False Positive Rate (probability of false alarm)
	NPV	   Negative Predictive Value
	PPV	   Positive Predictive Value (Precision)
	Prevalence    Prevalence
	TNR	   True Negative Rate (Specificity, Selectivity)
	TPR	   True Positive Rate (Recall, Sensitivity, probability of detection)
	TP	    True Positive
	FP	    False Positive
	FN	    False Negative
	TN	    True Negative
	TS	    Threat Score / Critical Success Index
	==========    ================================================================

	Input values must be positive and can be counts, percentage or decimal
	percentage.

	:attr float,int true_positive: True positive value
	:attr float,int false_positive: False positive value
	:attr float,int false_negative: False negative value
	:attr float,int true_negative: True negative value
	:attr float ACC: Accuracy
	:attr float F1score: F1 score
	:attr float FDR: False Discovery Rate
	:attr float FNR: False Negative Rate (Miss rate)
	:attr float FOR: False Omission Rate
	:attr float FPR: False Positive Rate (Probability of false alarm)
	:attr float NPV: Negative Predictive Value
	:attr float PPV: Positive Predictive Value (Precision)
	:attr float Prevalence: Prevalence
	:attr float TNR: True Negative Rate (Specificity, Selectivity)
	:attr float TPR: True Positive Rate (Recall, Sensitivity,
	Probability of detection)
	:attr float TS: Threat Score / Critical Success Index
	"""

	def __init__(
		self,
		truth: np.ndarray = None,
		prediction: np.ndarray = None,
		normalize: str = "all",
		labels_to_include: T.Iterable = None,
	):
		self.cm = confusion_matrix(truth, prediction, normalize=normalize, labels=labels_to_include)

		if labels_to_include is None:
			# labels are unique values in truth and prediction, sorted.
			self.labels = list(set(truth).union(set(prediction)))
		else:
			self.labels = tp.utils.ensure_list(labels_to_include)
		self.labels.sort()

		# If the shape of the confusion matrix is 2x2 then the binary statistics are available.
		# If not then 2 labels must be provided to yield a 2x2 matrix to enable the statistics.
		# Otherwise, only plotting is available.
		if self.cm.shape == (2, 2):
			self.TN, self.FP, self.FN, self.TP = self.cm.ravel()
			self._binary = True
		else:
			self.TN, self.FP, self.FN, self.TP = None, None, None, None
			self._binary = False

	def __repr__(self):
		if self._binary:
			return f"ConfusionMatrix(TP={self.TP}, FP={self.FP}, FN={self.FN}, TN={self.TN})"
		else:
			return f"ConfusionMatrix(shape={self.cm.shape})"

	@property
	def PPV(self):
		"""Positive Predictive Value (Precision)"""
		if self._binary:
			return self.TP / (self.TP + self.FP)
		else:
			logger.warning("PPV only available for binary classification.")

	@property
	def FDR(self):
		"""False Discovery Rate"""
		if self._binary:
			return self.FP / (self.TP + self.FP)
		else:
			logger.warning("FDR only available for binary classification.")

	@property
	def FOR(self):
		"""False Omission Rate"""
		if self._binary:
			return self.FN / (self.FN + self.TN)
		else:
			logger.warning("FOR only available for binary classification.")

	@property
	def NPV(self):
		"""Negative Predictive Value"""
		if self._binary:
			return self.TN / (self.FN + self.TN)
		else:
			logger.warning("NPV only available for binary classification.")

	@property
	def TPR(self):
		"""True Positive Rate (Recall, Sensitivity, Probability of detection)"""
		if self._binary:
			return self.TP / (self.TP + self.FN)
		else:
			logger.warning("TPR only available for binary classification.")

	@property
	def FNR(self):
		"""False Negative Rate (Miss rate)"""
		if self._binary:
			return self.FN / (self.FN + self.TN)
		else:
			logger.warning("FNR only available for binary classification.")

	@property
	def FPR(self):
		"""False Positive Rate (probability of false alarm)"""
		if self._binary:
			return self.FP / (self.FP + self.TN)
		else:
			logger.warning("FPR only available for binary classification.")

	@property
	def TNR(self):
		"""True Negative Rate (Specificity, Selectivity)"""
		if self._binary:
			return self.TN / (self.FP + self.TN)
		else:
			logger.warning("TNR only available for binary classification.")

	@property
	def Prevalence(self):
		"""Prevalence"""
		if self._binary:
			return (self.TP + self.FN) / (self.TP + self.FP + self.FN + self.TN)
		else:
			logger.warning("Prevalence only available for binary classification.")

	@property
	def ACC(self):
		"""Accuracy"""
		if self._binary:
			return (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)
		else:
			logger.warning("ACC only available for binary classification.")

	@property
	def F1score(self):
		"""F1 score"""
		if self._binary:
			return 2 * self.TP / (2 * self.TP + self.FP + self.FN)
		else:
			logger.warning("F1score only available for binary classification.")

	@property
	def TS(self):
		"""Threat Score / Critical Success Index"""
		if self._binary:
			return self.TP / (self.TP + self.FN + self.FP)
		else:
			logger.warning("TS only available for binary classification.")

	def plot_cm(
		self,
		figsize: tuple[int, int] = (6, 6),
		labels: list[str] = None,
		title: str = None,
		annotate: int | None = None,
		colorbar: bool = False,
		cbar_label: str = None,
		cmap: str = "viridis",
		title_fontdict: dict = None,
		tick_fontdict: dict = None,
		label_fontdict: dict = None,
		labelpad: int = 8,
		**kwargs,
	):
		"""
		Plot a confusion matrix as a heatmap.

		Args:
			figsize: Figure size.
			labels: List of labels for the confusion matrix. If given, length must match that of the confusion matrix.
			Default is the values in the data.
			title: Title of the confusion matrix plot.
			annotate: If None, no annotation is plotted. Otherwise the integer is the size of the annotated value
			in each cell.
			colorbar: If True, a colorbar is added to the plot.
			cmap: Colormap for the heatmap.
			tick_fontdict: Font dictionary for the axis labels.
			title_fontdict: Font dictionary for the title.
			kwargs: Additional keyword arguments for the seaborn heatmap function.

		Returns:
			fig, ax: Figure and axis objects.
		"""
		if labels is None:
			labels = self.labels

		if annotate is None:
			annotate = False
			annot_kws = None
		else:
			annot_kws = {"size": annotate}
			annotate = True

		fig, ax = plt.subplots(figsize=figsize)

		if colorbar:
			cbar = True
			cbar_kws = {"label": cbar_label}
		else:
			cbar = None
			cbar_kws = None

		if title_fontdict is None:
			title_fontdict = dict(size=12, weight="heavy")
		if tick_fontdict is None:
			tick_fontdict = dict(size=12, weight="medium")
		if label_fontdict is None:
			label_fontdict = dict(size=12, weight="medium")

		sns.heatmap(
			self.cm,
			ax=ax,
			annot=annotate,
			annot_kws=annot_kws,
			cmap=cmap,
			linewidths=1,
			linecolor="w",
			square=True,
			xticklabels=labels,
			yticklabels=labels,
			cbar=cbar,
			cbar_kws=cbar_kws,
			**kwargs,
		)

		ax.set_title(title, fontdict=title_fontdict)
		ax.set_xlabel("Truth", fontdict=label_fontdict, labelpad=labelpad)
		ax.set_ylabel("Predicted", fontdict=label_fontdict, labelpad=labelpad)
		ax.set_xticklabels(ax.get_xmajorticklabels(), fontdict=tick_fontdict)
		ax.set_yticklabels(ax.get_ymajorticklabels(), fontdict=tick_fontdict)
		fig.tight_layout()
		return fig, ax
