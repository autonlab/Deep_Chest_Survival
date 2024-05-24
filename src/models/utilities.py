import copy
import logging
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optimizer

from sksurv.linear_model.coxph import BreslowEstimator

import sys
sys.path.append('./')
from models.torch_models import DeepCoxPHTorchBottleneckTorch
from models.torch_models import DeepSurvivalMachinesBottleneckTorch
from models.torch_models import BinarySurvivalClassifierBottleneckTorch
from evaluation import evaluate

sys.path.append('./auton-survival/')
from auton_survival.models.dsm.losses import conditional_loss
from auton_survival.models.cph.dcph_utilities import partial_ll_loss
from auton_survival.models.dsm.utilities import pretrain_dsm


def bce_loss(logits, label, weight):
	criterion = nn.BCEWithLogitsLoss(reduction='none')
	mask = ~torch.isnan(label)

	loss = torch.zeros_like(label)
	loss[mask] = criterion(logits[mask], label[mask])
	loss = torch.nansum(loss * torch.from_numpy(weight).to(loss.device))

	return loss


def _save_model(model, train_loader, logging_filename):
	if isinstance(model, DeepCoxPHTorchBottleneckTorch):
		model.breslow_spline = _get_breslow_spline(model, train_loader)

	path = logging_filename+'_model' '.pkl'
	print("Saving the retrained model at", path)
	f = open(path, 'wb')
	pkl.dump(model, f)
	f.flush()
	f.close()


def _get_loss(model, x_image, x_demo, t, e, elbo=True, time=10*365.25):
	'''
	get loss for different models
	'''
	device = next(model.parameters()).device

	if isinstance(model, DeepSurvivalMachinesBottleneckTorch):
		srv_loss = conditional_loss(model, (x_image, x_demo), t, e, elbo=elbo) * x_image.shape[0]

	elif isinstance(model, BinarySurvivalClassifierBottleneckTorch):
		# get rid of event == 0, time-to-event < time
		mask = (e == 0) & (t < time)
		x_image = x_image[~mask].to(device).to(torch.float32)
		x_demo = x_demo[~mask].to(device).to(torch.float32)
		t = t[~mask].to(device).to(torch.float32)
		e = e[~mask].to(device).to(torch.float32)

		# positive class: time-to-event > time
		y = torch.zeros((x_image.shape[0]), dtype=torch.float).to(device)
		positive = (t > time)
		y[positive] = 1
		y = y.view(-1, 1)

		srv_loss = bce_loss(model((x_image, x_demo)), y, np.ones(1))

	elif isinstance(model, DeepCoxPHTorchBottleneckTorch):
		srv_loss = partial_ll_loss(model((x_image, x_demo)), t.detach().cpu().numpy(), e.detach().cpu().numpy())

	else:
		raise NotImplementedError()

	return srv_loss


def train_concept_bottleneck(model, train_loader, val_loader,
								eval_times=[1*365.25, 2*365.25, 5*365.25, 10*365.25],
								patience=3, epochs=20, lr=1e-3, elbo=True,
								random_seed=0, time=10*365.25,
								logging_filename='./dsm',):

	"""Function to train the torch instance of the model."""

	device = next(model.parameters()).device

	torch.manual_seed(random_seed)
	np.random.seed(random_seed)

	if val_loader is None:
		val_loader = train_loader

	model.train()

	logging.info('Pretraining the Underlying Distributions...')

	if isinstance(model, DeepSurvivalMachinesBottleneckTorch):
		premodel = pretrain_dsm(model,
								torch.tensor(train_loader.dataset.time.values),
								torch.tensor(train_loader.dataset.event.values),
								torch.tensor(val_loader.dataset.time.values),
								torch.tensor(val_loader.dataset.event.values),
								n_iter=10000, lr=1e-2, thres=1e-4)

		model.shape['1'].data.fill_(float(premodel.shape['1']))
		model.scale['1'].data.fill_(float(premodel.scale['1']))

	model.float()
	optim = optimizer.Adam(model.parameters(), lr=lr)

	patience_ = 0
	oldcost = float('inf')
	best_brier = float('inf')
	best_model_weights = None
	best_results = None

	for i in tqdm(range(epochs)):
		with open(logging_filename+'.txt', 'a') as f:
				print('epoch', i, '\n', file=f)

		# train
		model.train()
		tr_tot_loss = 0.

		for _, (x_image, x_demo, outcomes) in enumerate(tqdm(train_loader)):

			x_image = x_image.to(device).to(torch.float)
			x_demo = x_demo.to(device).to(torch.float)
			outcomes = outcomes.to(device).to(torch.float)

			t, e = outcomes[:, 0], outcomes[:, 1]

			optim.zero_grad()

			# get loss
			loss = _get_loss(model, x_image, x_demo, t, e, elbo, time)

			loss.backward()
			optim.step()

			tr_tot_loss += loss.detach().cpu().numpy()

		with open(logging_filename+'.txt', 'a') as f:
			print('train_tot_loss', float(tr_tot_loss), '\n', file=f)

		# validation
		vl_tot_loss = 0.
		model.eval()
		for _, (x_image, x_demo, outcomes) in enumerate(tqdm(val_loader)):

			with torch.no_grad():
				x_image = x_image.to(device).to(torch.float)
				x_demo = x_demo.to(device).to(torch.float)
				t = outcomes[:, 0].to(device).to(torch.float)
				e = outcomes[:, 1].to(device).to(torch.float)

				vl_tot_loss += _get_loss(model, x_image, x_demo, t, e, elbo, time)

		vl_tot_loss = float(vl_tot_loss)

		with open(logging_filename+'.txt', 'a') as f:
			print('val_tot_loss', vl_tot_loss, '\n', file=f)

		if isinstance(model, DeepCoxPHTorchBottleneckTorch):
			model.breslow_spline = _get_breslow_spline(model, train_loader)

		results = evalute_model(model, val_loader, train_loader, eval_times, n_bootstrap=None)
		with open(logging_filename+'.txt', 'a') as f:
			print(results, '\n\n', file=f)

		# save model
		current_brier = np.mean(results['Brier Score'])
		if current_brier < best_brier:
			best_brier = current_brier
			best_model_weights = copy.deepcopy(model.state_dict())
			best_results = results

		if current_brier >= oldcost:
			if patience_ == patience:
				model.load_state_dict(best_model_weights)
				_save_model(model, train_loader, logging_filename)

				with open(logging_filename+'.txt', 'a') as f:
					print('Best val results', best_results, '\n', file=f)

				return model

			else:
				patience_ += 1

		else:
			patience_ = 0

		oldcost = current_brier

	model.load_state_dict(best_model_weights)
	_save_model(model, train_loader, logging_filename)

	with open(logging_filename+'.txt', 'a') as f:
		print('Best val results', best_results, '\n', file=f)

	return model



def _get_breslow_spline(model, train_loader):
	'''
	get fitted breslow spline
	'''
	preds_all = []

	device = next(model.parameters()).device

	for _, (x_image, x_demo, _) in enumerate(tqdm(train_loader)):
		with torch.no_grad():

			x_image = x_image.to(device).to(torch.float)
			x_demo = x_demo.to(device).to(torch.float)

			logits = model((x_image, x_demo))

			preds_all.append(logits.detach().cpu().ravel())

	breslow_spline = BreslowEstimator().fit(np.concatenate(preds_all),
											train_loader.dataset.event.values,
											train_loader.dataset.time.values)

	return breslow_spline





def _predict_survival(model, x, times):
	'''
	get survival probabilities
	'''
	if isinstance(model, BinarySurvivalClassifierBottleneckTorch):
  		return model(x).detach().cpu().numpy().ravel()

	elif isinstance(model, DeepCoxPHTorchBottleneckTorch):
		from auton_survival.models.cph.dcph_utilities import predict_survival
		return predict_survival((model, model.breslow_spline), x, times)

	elif isinstance(model, DeepSurvivalMachinesBottleneckTorch):
		from auton_survival.models.dsm.losses import predict_cdf
		scores = predict_cdf(model, x, times)
		return np.exp(np.array(scores)).T

	else:
		raise NotImplementedError()



def evalute_model(model, test_loader, train_loader,
					times=[1 * 365.25, 2 * 365.25, 5 * 365.25, 10 * 365.25],
					n_bootstrap=None,):
	'''
	performance with test set
	'''
	model = model.eval()

	device = next(model.parameters()).device

	predictions_te = []

	for _, (x_image, x_demo, _) in enumerate(tqdm(test_loader)):

		x_image = x_image.to(torch.float).to(device)
		x_demo = x_demo.to(torch.float).to(device)

		with torch.no_grad():
			preds = _predict_survival(model, (x_image, x_demo), times)
			predictions_te.append(preds)

	predictions_te = np.concatenate(predictions_te, axis=0)
	y_te = np.concatenate((np.reshape(test_loader.dataset.time.values, (-1, 1)),
                        np.reshape(test_loader.dataset.event.values, (-1, 1))), axis=1)
	y_te = pd.DataFrame(y_te, columns=['time', 'event'])

	y_tr = np.concatenate((np.reshape(train_loader.dataset.time.values, (-1, 1)),
                        np.reshape(train_loader.dataset.event.values, (-1, 1))), axis=1)
	y_tr = pd.DataFrame(y_tr, columns=['time', 'event'])

	results = evaluate(y_te, predictions_te, y_tr, times=times, random_seed=0,
                    n_bootstrap=n_bootstrap, binary=isinstance(model, BinarySurvivalClassifierBottleneckTorch))

	return results
