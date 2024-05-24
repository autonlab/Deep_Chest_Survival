import numpy as np

import torch
from torch import nn

import sys
sys.path.append('./auton-survival/')
sys.path.append('/home/scratch/mingzhul/generative_lung_survival/auton-survival/')
sys.path.append('/home/scratch/mingzhul/generative_lung_survival/torchxrayvision/')
sys.path.append('/zfsauton2/home/mingzhul/generative_lung_survival/torchxrayvision/')

from auton_survival.models.cph.dcph_torch import DeepCoxPHTorch
from auton_survival.models.dsm.dsm_torch import DeepSurvivalMachinesTorch
import auton_survival.models.dsm.losses as losses

import torchxrayvision as xrv



class DeepCoxPHTorchBottleneckTorch(DeepCoxPHTorch):

	"""
	inputdim: int
		Dimensionality of the input features.
	layers: list
		A list of integers consisting of the number of neurons in each
		hidden layer.
	"""

	def __init__(self, demo_size, layers=[], optimizer='Adam', 
				 image_embedding_dim=15):

		super(DeepCoxPHTorchBottleneckTorch,
			self).__init__(inputdim=image_embedding_dim+demo_size,
							layers=layers, optimizer=optimizer)

		image_embedding = xrv.models.DenseNet(weights='densenet121-res224-all')
		image_embedding.classifier = nn.Linear(image_embedding.classifier.in_features, 15)
		image_embedding.op_threshs = None
		image_embedding.apply_sigmoid = False
		self.image_embedding = image_embedding

	def bottleneck(self, x_image):

		bs, n_crops, c, h, w = x_image.size()
		x_image_rep = self.image_embedding(x_image.view(-1, c, h, w))
		x_image_rep = x_image_rep.view(bs, n_crops, -1).mean(dim=1)

		return x_image_rep

	def forward(self, x):
		"""The forward function that is called when data is passed through DCPH.
		Args:
		x:
			a torch.tensor of the input images.
		"""
		x_image, x_demo = x
		x_image_rep = self.bottleneck(x_image)
		x_image_pred = torch.sigmoid(x_image_rep)
		xrep = torch.cat([x_image_pred, x_demo], dim=1)

		return self.expert(self.embedding(xrep))


class BinarySurvivalClassifierBottleneckTorch(DeepCoxPHTorchBottleneckTorch):
	"""
	inputdim: int
		Dimensionality of the input features.
	layers: list
		A list of integers consisting of the number of neurons in each
		hidden layer.
	"""

	def __init__(self, demo_size, layers=[], optimizer='Adam', image_embedding_dim=15):

		super(BinarySurvivalClassifierBottleneckTorch,
			self).__init__(demo_size=demo_size,
							layers=layers,
							optimizer=optimizer,
							image_embedding_dim=image_embedding_dim,)


	# Binary Classifier requires bias in the output, unlike CoxPH model.
	def _init_coxph_layers(self, lastdim):
		self.expert = nn.Linear(lastdim, 1, bias=True)




class DeepSurvivalMachinesBottleneckTorch(DeepSurvivalMachinesTorch):

	"""
	inputdim: int
		Dimensionality of the input features.
	k: int
		The number of underlying parametric distributions.
	layers: list
		A list of integers consisting of the number of neurons in each
		hidden layer.
	init: tuple
		A tuple for initialization of the parameters for the underlying
		distributions. (shape, scale).
	activation: str
		Choice of activation function for the MLP representation.
		One of 'ReLU6', 'ReLU' or 'SeLU'.
		Default is 'ReLU6'.
	dist: str
		Choice of the underlying survival distributions.
		One of 'Weibull', 'LogNormal'.
		Default is 'Weibull'.
	temp: float
		The logits for the gate are rescaled with this value.
		Default is 1000.
	discount: float
		a float in [0,1] that determines how to discount the tail bias
		from the uncensored instances.
		Default is 1.
	"""

	def __init__(self, k, demo_size, layers=[], dist='Weibull', temp=1000.,
				discount=1.0, optimizer='Adam', image_embedding_dim=15):

		super(DeepSurvivalMachinesBottleneckTorch,
			self).__init__(k=k, inputdim=image_embedding_dim+demo_size,
							layers=layers, dist=dist, temp=temp,
							discount=discount, optimizer=optimizer)

		image_embedding = xrv.models.DenseNet(weights='densenet121-res224-all')
		image_embedding.classifier = nn.Linear(image_embedding.classifier.in_features, 15)
		image_embedding.op_threshs = None
		image_embedding.apply_sigmoid = False
		self.image_embedding = image_embedding


	def bottleneck(self, x_image):

		bs, n_crops, c, h, w = x_image.size()
		x_image_rep = self.image_embedding(x_image.view(-1, c, h, w))
		x_image_rep = x_image_rep.view(bs, n_crops, -1).mean(dim=1)

		return x_image_rep

	def forward(self, x, risk='1'):
		"""The forward function that is called when data is passed through DSM.
		Args:
		x:
			a torch.tensor of the input images.
		"""

		x_image, x_demo = x
		x_image_rep = self.bottleneck(x_image)
		x_image_pred = torch.sigmoid(x_image_rep)
		xrep = torch.cat([x_image_pred, x_demo], dim=1)
		xrep = self.embedding(xrep)

		dim = xrep.shape[0]

		dsm_output = (self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
					self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
					self.gate[risk](xrep)/self.temp)

		return dsm_output

	def predict_pdf(self, x, t, risk=1):

		if not isinstance(t, list):
			t = [t]

		scores = losses.predict_pdf(self, x, t, risk=str(risk))
		return np.exp(np.array(scores)).T









