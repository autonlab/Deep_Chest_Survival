import os
import numpy as np
from PIL import Image
import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

from datasets import load_plco_trial_data

from torchvision import transforms

import sys
sys.path.append('./auton-survival/')
from auton_survival.preprocessing import Preprocessor



class ChestXrayDataset(Dataset):

	def __init__(self,
				preprocessor=None,
				metadata_dir='datasets/package-plco/Lung/',
				img_dir='datasets/imgs',
				transform=None,
				sample='train',
				random_seed=0,
				return_image_names=False,
				metadata=None):

		super(ChestXrayDataset, self).__init__()

		self.random_seed = random_seed
		self.img_dir = img_dir
		self.transform = transform
		self.images = []
		self.count = 0
		self.sample = sample
		self.return_image_names = return_image_names

		print(metadata_dir, img_dir)

		assert sample in ['test', 'train', 'val']

		if metadata is None:
			metadata = load_plco_trial_data(metadata_dir)
			if sample == 'test':
				metadata = metadata[metadata.cv_folds == 4]
			elif sample == 'val':
				metadata = metadata[metadata.cv_folds == 3]
			elif sample == 'train':
				mask = (metadata.cv_folds==3)|(metadata.cv_folds==4)
				metadata = metadata[~mask]
			else:
				raise ValueError('sample must be one of test, val, train')

		self.metadata = metadata

		cat_feats = ['sex', 'race7', 'hispanic_f', 'cig_stat']
		num_feats = ['age']

		if preprocessor is not None:
			self.demo = preprocessor.fit_transform(self.metadata[cat_feats+num_feats],
													cat_feats=cat_feats,
													num_feats=num_feats,
													one_hot=True, fill_value=-1)
		else:
			self.demo = self.metadata[cat_feats+num_feats]

		self.demo_shape = len(self.demo.iloc[0].values)

		self.metadata = shuffle(self.metadata, random_state=self.random_seed)

		self.time = self.metadata['time']
		self.event = self.metadata['event']


	def __getitem__(self, idx):

		image_name, batch_name = self.metadata.index[idx]
		path = os.path.join(self.img_dir, 'batch_' + batch_name.lower(),
							image_name)
		image = Image.open(path)

		time = self.time.loc[image_name, batch_name].copy()
		event = self.event.loc[image_name, batch_name].copy()
		time_to_event = torch.tensor([time, event])
		demo = self.demo.loc[image_name, batch_name].values.copy().astype(np.float32)

		if self.transform is not None:
			image = self.transform(image)

		if self.return_image_names:
			return self.metadata.index[idx], image, demo, time_to_event

		else:
			return image, demo, time_to_event


	def get_feature_names(self):
		return self.feature_names

	def get_demo_names(self):
		return list(self.demo)

	def __len__(self):
		return self.metadata.shape[0]


def instantiate_data_loader(sample='train',
							batch_size=64,
							random_seed=0,
							img_dir='datasets/imgs',
							metadata_dir='datasets/package-plco/Lung/',
							inference=False,
							return_image_names=True,
							metadata=None):

	print("Instantiating PLCO Loader...")

	if inference:
		transform = transforms.Compose([
										transforms.FiveCrop(224),
										transforms.Lambda
										(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
	else:
		transform = transforms.Compose([
										transforms.RandomCrop(224),
										transforms.Lambda
										(transforms.ToTensor()),
										transforms.Lambda
										(lambda crop: torch.unsqueeze(crop, 0))])

	preprocessor = Preprocessor(cat_feat_strat='replace', num_feat_strat='mean')

	dataset = ChestXrayDataset(preprocessor=preprocessor,
								transform=transform,
								img_dir=img_dir,
								random_seed=random_seed,
								sample=sample,
								metadata_dir=metadata_dir,
								return_image_names=return_image_names,
								metadata=metadata)

	loader = DataLoader(dataset=dataset, batch_size=batch_size,
						shuffle=False, num_workers=128, pin_memory=True)

	return loader

