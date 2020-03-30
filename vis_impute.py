from argparse import ArgumentParser
from copy import deepcopy
from importlib import import_module
from math import ceil
from os.path import exists, join
from sys import stderr

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import compute_normalization
from imputation_networks import get_imputation_networks
from train_utils import extend_batch, get_validation_iwae
from VAEAC import VAEAC


class ArgParseRange:
    """
    List with this element restricts the argument to be
    in range [start, end].
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __repr__(self):
        return '{0}...{1}'.format(self.start, self.end)


parser = ArgumentParser(description='Missing Features Multiple Imputation.')

parser.add_argument('--input_file', type=str, action='store', required=True,
                    help='Path to the input .tsv file. ' +
                         'NaNs are missing values.')

parser.add_argument('--output_file', type=str, action='store', required=True,
                    help='Path to the output .tsv file.')

parser.add_argument('--num_imputations', type=int, action='store', default=5,
                    help='Number of imputations to generate per object. ' +
                         'Default: 5.')

parser.add_argument('--one_hot_max_sizes', action='store', type=int,
                    nargs='+', required=True,
                    help='The space-separated list of one-hot max sizes ' +
                         'for categorical features and 0 or 1 ' +
                         'for real-valued ones. A categorical feature ' +
                         'is supposed to be a column of integers ' +
                         'from 0 to K-1, where K is one-hot max size ' +
                         'for the feature. The length of the list ' +
                         'must be equal to the number of columns ' +
                         'in the data.')

args = parser.parse_args()

one_hot_max_sizes = args.one_hot_max_sizes

# Read and normalize input data
raw_data = np.loadtxt(args.input_file, delimiter='\t')
raw_data = torch.from_numpy(raw_data).float()

norm_mean, norm_std = compute_normalization(raw_data, one_hot_max_sizes)
norm_std = torch.max(norm_std, torch.tensor(1e-9))
data = (raw_data - norm_mean[None]) / norm_std[None]

# data = raw_data

# Default parameters which are not supposed to be changed from user interface
use_cuda = torch.cuda.is_available()
verbose = True
# Non-zero number of workers cause nasty warnings because of some bug in
# multiprocess library. It might be fixed now, but anyway there is no need
# to have a lot of workers for dataloader over in-memory tabular data.
num_workers = 0

# design all necessary networks and learning parameters for the dataset
networks = get_imputation_networks(one_hot_max_sizes)

# build VAEAC on top of returned network, optimizer on top of VAEAC,
# extract optimization parameters and mask generator
model = VAEAC(
    networks['reconstruction_log_prob'],
    networks['proposal_network'],
    networks['prior_network'],
    networks['generative_network']
)
if use_cuda:
    model = model.cuda()
optimizer = networks['optimizer'](model.parameters())
batch_size = networks['batch_size']
mask_generator = networks['mask_generator']
vlb_scale_factor = networks.get('vlb_scale_factor', 1)

# checkpoint = torch.load('state_full')
checkpoint = torch.load('29_3_150_0.1_denormal')

model.load_state_dict(checkpoint['model_state_dict'])

# build dataloader for the whole input data
dataloader = DataLoader(data, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        drop_last=False)


# prepare the store for the imputations
results = []
for i in range(args.num_imputations):
    results.append([])

iterator = dataloader
if verbose:
    iterator = tqdm(iterator)

# impute missing values for all input data
for batch in iterator:

    # if batch size is less than batch_size, extend it with objects
    # from the beginning of the dataset
    batch_extended = torch.tensor(batch)
    batch_extended = extend_batch(batch_extended, dataloader, batch_size)

    if use_cuda:
        batch = batch.cuda()
        batch_extended = batch_extended.cuda()

    # compute the imputation mask
    mask_extended = torch.isnan(batch_extended).float()

    # compute imputation distributions parameters
    with torch.no_grad():
        samples_params = model.generate_samples_params(batch_extended,
                                                       mask_extended,
                                                       args.num_imputations)
        samples_params = samples_params[:batch.shape[0]]

    # make a copy of batch with zeroed missing values
    mask = torch.isnan(batch)
    batch_zeroed_nans = torch.tensor(batch)
    batch_zeroed_nans[mask] = 0

    # impute samples from the generative distributions into the data
    # and save it to the results
    for i in range(args.num_imputations):
        sample_params = samples_params[:, i]
        sample = networks['sampler'](sample_params)
        sample[(1 - mask).byte()] = 0
        sample += batch_zeroed_nans
        results[i].append(torch.tensor(sample, device='cpu'))

# concatenate all batches into one [n x K x D] tensor,
# where n in the number of objects, K is the number of imputations
# and D is the dimensionality of one object
for i in range(len(results)):
    results[i] = torch.cat(results[i]).unsqueeze(1)
result = torch.cat(results, 1)

# reshape result, undo normalization and save it
result = result.view(result.shape[0] * result.shape[1], result.shape[2])
result = result * norm_std[None] + norm_mean[None]

np.savetxt(args.output_file, result.numpy(), delimiter='\t')
