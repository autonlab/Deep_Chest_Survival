import torch
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.append('./auton-survival/')
from auton_survival.metrics import survival_regression_metric

from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter



def compute_AUCs(gt, pred):
	"""Computes Area Under the Curve (AUC) from prediction scores.

	Args:
		gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
			true binary labels.
		pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
			can either be probability estimates of the positive class,
			confidence values, or binary decisions.

	Returns:
		List of AUROCs of all classes.
	"""

	AUROCs = []
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()
	for i in range(gt_np.shape[1]):
		# ignore nan
		gts = gt_np[:, i][~np.isnan(gt_np[:, i])]
		preds = pred_np[:, i][~np.isnan(gt_np[:, i])]

		if sum(gts) == len(gts) or sum(gts) == 0:
			AUROCs.append(float("nan"))
		else:
			AUROCs.append(roc_auc_score(gts, preds))
	return torch.tensor(AUROCs)



def _calibration_curve_km(out,
							e,
							t,
							a,
							group,
							eval_time,
							ret_bins=True,
							strat='quantile',
							n_bins=10,):

	"""Returns the Calibration curve and the bins given some risk scores.
	Accepts the output of a trained survival model at a certain evaluation time,
	the event indicators and protected group membership and outputs an KM
	adjusted calibration curve.
	Args:
		out:
			risk scores P(T>t) issued by a trained survival analysis model
			(output of deep_cox_mixtures.models.predict_survival).
		e:
			a numpy vector of indicators specifying is event or censoring occured.
		t:
			a numpy vector of times at which the events or censoring occured.
		a:
			a numpy vector of protected attributes.
		group:
			string indicating the demogrpahic to evaluate calibration for.
		eval_time:
			float/int of the event time at which calibration is to be evaluated. Must
			be same as the time at which the Risk Scores were issues.
		ret_bins:
			Boolean that specifies if the bins of the calibration curve are to be
			returned.
		strat:
			Specifies how the bins are computed. One of:
			"quantile": Equal sized bins.
			"uniform": Uniformly stratified.
		n_bins:
			int specifying the number of bins to use to compute the ece.
	Returns:
		Calibration Curve: A tuple of True Probality, Estimated Probability in
		each bin and the estimated Expected Calibration Error.
	"""

	out_ = out.copy()

	if group is not None:
		e = e[a == group]
		t = t[a == group]
		out = out[a == group]

	y = t > eval_time

	if strat == 'quantile':

		quantiles = [(1. / n_bins) * i for i in range(n_bins + 1)]
		outbins = np.quantile(out, quantiles)

	if strat == 'uniform':

		binlen = (out.max() - out.min()) / n_bins
		outbins = [out.min() + i * binlen for i in range(n_bins + 1)]

	prob_true = []
	prob_pred = []

	ece = 0

	for n_bin in range(n_bins):

		binmin = outbins[n_bin]
		binmax = outbins[n_bin + 1]

		scorebin = (out >= binmin) & (out <= binmax)

		weight = float(scorebin.sum()) / len(out)

		out_ = out[scorebin]
		y_ = y[scorebin]

		if len(t[scorebin]) == 0:
			prob_true.append(np.nan)
			prob_pred.append(np.nan)
			continue
		
		pred = KaplanMeierFitter().fit(t[scorebin], e[scorebin]).predict(eval_time)
		prob_true.append(pred)
		prob_pred.append(out_.mean())

		gap = abs(prob_pred[-1] - prob_true[-1])

		ece += weight * gap

	if ret_bins:
		return prob_true, prob_pred, outbins, ece

	else:
		return prob_true, prob_pred, ece



def calibration_curve(out,
						e,
						t,
						a,
						group,
						eval_time,
						typ='KM',
						ret_bins=False,
						strat='quantile',
						n_bins=25,
						random_seed=None,):
	"""Returns the Calibration curve and the bins given some risk scores.
	Accepts the output of a trained survival model at a certain evaluation time,
	the event indicators and protected group membership and outputs a calibration
	curve
	Args:
		out:
			risk scores P(T>t) issued by a trained survival analysis model
			(output of deep_cox_mixtures.models.predict_survival).
		e:
			a numpy vector of indicators specifying is event or censoring occured.
		t:
			a numpy vector of times at which the events or censoring occured.
		a:
			a numpy vector of protected attributes.
		group:
			string indicating the demogrpahic to evaluate calibration for.
			use None for entire population.
		eval_time:
			float/int of the event time at which calibration is to be evaluated. Must
			be same as the time at which the Risk Scores were issued.
		typ:
			Determines if the calibration curves are to be computed on the individuals
			that experienced the event or adjusted estimates for individuals that are
			censored using IPCW estimator on a population or subgroup level
		ret_bins:
			Boolean that specifies if the bins of the calibration curve are to be
			returned.
		strat:
			Specifies how the bins are computed. One of:
			"quantile": Equal sized bins.
			"uniform": Uniformly stratified.
		n_bins:
			int specifying the number of bins to use to compute the ece.
	Returns:
		Calibration Curve: A tuple of True Probality and Estimated Probability in
		each bin.
	"""

	idx = np.arange(len(out))
	if random_seed is not None:
		np.random.seed(random_seed)
		idx = np.random.choice(idx, len(out), replace=True)

	return _calibration_curve_km(out[idx],
									e[idx],
									t[idx],
									a,
									group,
									eval_time,
									ret_bins=ret_bins,
									strat=strat,
									n_bins=n_bins,)


def get_ece(times, preds, e, t, n_bootstrap=None):
	eces = []
	for i, eval_time in enumerate(times):
		if n_bootstrap is None:
			_, _, _, ece = calibration_curve(out=preds[:, i],
												e=e,
												t=t,
												a=None,
												group=None,
												eval_time=eval_time,
												typ='KM',
												ret_bins=True,
												strat='quantile',
												n_bins=25,)
			eces.append(ece)

		else:
			ece = np.array([calibration_curve(out=preds[:, i],
												e=e,
												t=t,
												a=None,
												group=None,
												eval_time=eval_time,
												typ='KM',
												ret_bins=True,
												strat='quantile',
												n_bins=25, random_seed=j)[3] for j in range(n_bootstrap)])
			eces.append(ece)

	return eces







def evaluate(y_te, predictions_te, y_tr,
				times=[5 * 365.25, 10 * 365.25, 15 * 365.25], 
				n_bootstrap=None,
				random_seed=0,
				binary=False):
	'''
	evaluate the performance with test set
	'''
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)

	results = dict()
	if binary:
		mask = (y_te['event'] == 0) & (y_te['time'] < times[0])
		e = y_te['event'][~mask]
		t = y_te['time'][~mask]

		mask = mask.to_numpy()
		predictions_te = predictions_te[~mask].reshape(-1)
		# sigmoid
		predictions_te = 1 / (1 + np.exp(-predictions_te))

		y = np.zeros_like(e.to_numpy())
		positive = (t > times[0])
		y[positive] = 1

		from sklearn import metrics
		
		if n_bootstrap is None:
			fpr, tpr, thresholds = metrics.roc_curve(y, predictions_te, pos_label=1)
			auc = metrics.auc(fpr, tpr)

			concordance_index = np.nan

			brier_score = metrics.brier_score_loss(y, predictions_te, pos_label=1)

			from torchmetrics.classification import BinaryCalibrationError
			m = BinaryCalibrationError(n_bins=50, norm='l1')
			ece = m(torch.from_numpy(predictions_te), torch.from_numpy(y)).item()

		else:
			auc = []
			brier_score = []
			ece = []
			
			concordance_index = survival_regression_metric('ctd', y_te[~mask],
															predictions_te[:, None],
															times,
															outcomes_train=y_tr,
															n_bootstrap=n_bootstrap)
			
			for j in range(n_bootstrap):
				np.random.seed(j)

				idx = np.arange(len(y))
				idx = np.random.choice(idx, len(y), replace=True)

				y_ = y[idx]
				pred_ = predictions_te[idx]		 

				fpr, tpr, thresholds = metrics.roc_curve(y_, pred_, pos_label=1)
				auc.append(metrics.auc(fpr, tpr))

				brier_score.append(metrics.brier_score_loss(y_, pred_, pos_label=1))

				from torchmetrics.classification import BinaryCalibrationError
				m = BinaryCalibrationError(n_bins=50, norm='l1')
				ece.append(m(torch.from_numpy(pred_), torch.from_numpy(y_)).item())

	else:
		brier_score = survival_regression_metric('brs', y_te,
													predictions_te, times,
													outcomes_train=y_tr,
													n_bootstrap=n_bootstrap)
		concordance_index = survival_regression_metric('ctd', y_te,
														predictions_te,
														times,
														outcomes_train=y_tr,
														n_bootstrap=n_bootstrap)
		auc = survival_regression_metric('auc', y_te, predictions_te,
											times, outcomes_train=y_tr,
											n_bootstrap=n_bootstrap)

		ece = get_ece(times, predictions_te, y_te['event'], y_te['time'],
						n_bootstrap=n_bootstrap)		

	if n_bootstrap is None:
		results['Brier Score'] = brier_score
		results['Concordance Index'] = concordance_index
		results['AUC'] = auc
		results['ECE'] = ece
 
	else:
		results['Brier Score'] = dict()
		results['Concordance Index'] = dict()
		results['AUC'] = dict()
		results['ECE'] = dict()

		results['Brier Score']['lower'] = np.quantile(np.array(brier_score), 0.025, axis=0)
		results['Brier Score']['upper'] = np.quantile(np.array(brier_score), 0.975, axis=0)
		results['Concordance Index']['lower'] = np.quantile(np.array(concordance_index), 0.025, axis=0)
		results['Concordance Index']['upper'] = np.quantile(np.array(concordance_index), 0.975, axis=0)
		results['AUC']['lower'] = np.quantile(np.array(auc), 0.025, axis=0)
		results['AUC']['upper'] = np.quantile(np.array(auc), 0.975, axis=0)
		results['ECE']['lower'] = np.quantile(np.transpose(np.array(ece)), 0.025, axis=0)
		results['ECE']['upper'] = np.quantile(np.transpose(np.array(ece)), 0.975, axis=0)
		

		brier = (np.array(results['Brier Score']['lower']) + np.array(results['Brier Score']['upper'])) / 2
		ci = (np.array(results['Concordance Index']['lower']) + np.array(results['Concordance Index']['upper'])) / 2
		auc = (np.array(results['AUC']['lower']) + np.array(results['AUC']['upper'])) / 2
		ece = (np.array(results['ECE']['lower']) + np.array(results['ECE']['upper'])) / 2

		results['Brier Score']['median'] = brier
		results['Concordance Index']['median'] = ci
		results['AUC']['median'] = auc
		results['ECE']['median'] = ece

	return results
