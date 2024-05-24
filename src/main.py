import io
import copy
import torch
import numpy as np

import os
import argparse
import pickle as pkl

from sklearn.model_selection import ParameterGrid

import sys
sys.path.append('./')
sys.path.append('./auton-survival/')

from image_datasets import instantiate_data_loader
from models.utilities import train_concept_bottleneck, evalute_model
from models.torch_models import DeepCoxPHTorchBottleneckTorch
from models.torch_models import BinarySurvivalClassifierBottleneckTorch
from models.torch_models import DeepSurvivalMachinesBottleneckTorch




# TODO: implement no_km_binary in auton-survival/auton_survival/models/binary/utilities.py


# tmux 0: cph. tmux 6: bin
random_seed = 0
img_dir = '/zfsauton2/home/mingzhul/generative_lung_survival/datasets/preprocessed_images'
metadata_dir = '/zfsauton2/home/mingzhul/generative_lung_survival/datasets/package-plco/Lung/'
model_name = 'cph'
n_bootstrap = 100
tr_batch_size = 128
vl_batch_size = 64
epoch = 10
patience = 3
lr = 3e-4
eval_time_horizon = 10*365.25
image_embedding_dim = 15


device = 'cuda:0'
if model_name == 'bin':
    eval_times = [5 * 365.25]
if model_name == 'cph':
    eval_times = [2 * 365.25, 5 * 365.25, 10 * 365.25]
    
# all models
layer = [[64, 64]]
# DSM
temp = [1]
discount = [1]
k = [4]
elbo = [True]





if model_name == 'bin':
    assert(len(eval_times) == 1)
    eval_time_horizon = eval_times[0]


torch.manual_seed(random_seed)
np.random.seed(random_seed)

hyperparams = {}

if model_name == 'dsm':
    hyperparams['temp'] = temp
    hyperparams['k'] = k
    hyperparams['discount'] = discount
    hyperparams['elbo'] = elbo

if model_name in ['dsm', 'cph', 'bin']:
    hyperparams['layer'] = layer

else:
    raise NotImplementedError('No model named', model_name)


# Instatiate the Train, Val and Test Image Data Loaders.
train_loader = instantiate_data_loader(sample='train', batch_size=tr_batch_size,
                                        img_dir=img_dir, metadata_dir=metadata_dir,
                                        inference=False, return_image_names=False,
                                        random_seed=random_seed)
val_loader = instantiate_data_loader(sample='val', batch_size=vl_batch_size,
                                    img_dir=img_dir, metadata_dir=metadata_dir,
                                    inference=True, return_image_names=False,
                                    random_seed=random_seed)
test_loader = instantiate_data_loader(sample='test', batch_size=vl_batch_size,
                                    img_dir=img_dir, metadata_dir=metadata_dir,
                                    inference=True, return_image_names=False,
                                    random_seed=random_seed)




class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


with open("trained_models/bottleneck/bin/[1826.25]/{'layer': [64, 64], 'eval_times': [1826.25]}_model.pkl",'rb') as f:
    model = CPU_Unpickler(f).load()
model = model.to(device)
    
results = evalute_model(model, test_loader, train_loader, [eval_times[1]], n_bootstrap=n_bootstrap)
print('Test results', results)


with open("trained_models/bottleneck/cph/[730.5, 1826.25, 3652.5]/{'layer': [64, 64], 'eval_times': [730.5, 1826.25, 3652.5]}_model.pkl",'rb') as f:
    model = CPU_Unpickler(f).load()
model = model.to(device)
    
results = evalute_model(model, test_loader, train_loader, eval_times, n_bootstrap=n_bootstrap,)
print('Test results', results)




print(bvsd)






lowest_bs = np.inf

# Loop over each models Hyperparameter Grid.
for parameters in ParameterGrid(hyperparams):
    print(parameters)
    if model_name == 'cph':
        model = DeepCoxPHTorchBottleneckTorch(demo_size=train_loader.dataset.demo_shape,
                                              layers=parameters['layer'],
                                              optimizer='Adam',
                                              image_embedding_dim=image_embedding_dim,)
    elif model_name == 'dsm':
        model = DeepSurvivalMachinesBottleneckTorch(k=parameters['k'],
                                                    demo_size=train_loader.dataset.demo_shape,
                                                    layers=parameters['layer'],
                                                    temp=parameters['temp'],
                                                    discount=parameters['discount'],
                                                    image_embedding_dim=image_embedding_dim,)
    elif model_name == 'bin':
        model = BinarySurvivalClassifierBottleneckTorch(demo_size=train_loader.dataset.demo_shape,
                                                        layers=parameters['layer'],
                                                        optimizer='Adam',
                                                        image_embedding_dim=image_embedding_dim,)

    else:
        raise NotImplementedError('no model named', model_name)

    # train each model and evaluate performance on test dataset
    logging_filename = './trained_models/bottleneck/' + model_name + '/' + str(eval_times) + '/'

    if not os.path.exists(logging_filename):
        os.makedirs(logging_filename)

    logging_filename += str(parameters | {'eval_times': eval_times})
    model = model.to(device)

    model = train_concept_bottleneck(model, train_loader, val_loader,
                                     random_seed=random_seed,
                                     elbo=parameters['elbo'] if 'elbo' in parameters else None,
                                     lr=lr,
                                     epochs=epoch,
                                     patience=patience,
                                     time=eval_time_horizon,
                                     eval_times=eval_times,
                                     logging_filename=logging_filename)

    # val
    results = evalute_model(model, val_loader, train_loader, eval_times, n_bootstrap=None,)
    bs = np.mean(results['Brier Score'])
    if bs < lowest_bs:
        lowest_bs = bs
        best_model_weights = copy.deepcopy(model.state_dict())


model.load_state_dict(best_model_weights)
results = evalute_model(model, test_loader, train_loader, eval_times, n_bootstrap=n_bootstrap,)

with open(logging_filename+'.txt', 'a') as f:
    print('Test results', results, '\n\n', file=f)

