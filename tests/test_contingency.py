import numpy as np

from terrapyn.scoring import ConfusionMatrix


def test_confusion_matrix():
	truth = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0])
	prediction = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 0])
	cm = ConfusionMatrix(truth, prediction)
	assert cm.TP == 0.2
	assert cm.FP == 0.3
	assert cm.FN == 0.2
	assert cm.TN == 0.3
	assert cm.ACC == 0.5
	assert cm.F1score == 0.44444444444444453
	assert cm.FDR == 0.6
	assert cm.FNR == 0.4
	assert cm.FOR == 0.4
	assert cm.FPR == 0.5
	assert cm.NPV == 0.6
	assert cm.PPV == 0.4
	assert cm.Prevalence == 0.4
	assert cm.TNR == 0.5
	assert cm.TPR == 0.5
	assert cm.TS == 0.28571428571428575
