from os import listdir
# from sys import argv
from argparse import ArgumentParser

import numpy as np

def load_data(filename):
    return np.loadtxt(filename, delimiter='\t')

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

parser.add_argument('--groundtruth', type=str, action='store', required=True)

parser.add_argument('--input_file', type=str, action='store', required=True,
                    help='Path to the input .tsv file. ' +
                         'NaNs are missing values.')

parser.add_argument('--imputed_file', type=str, action='store', required=True)

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

groundtruth = load_data(args.groundtruth)
input_data = load_data(args.input_file)
output_data = load_data(args.imputed_file)

def compute_nrmse(gt, mask, imputations):
    # Compute normalized root mean squared error for a column.
    std = gt.std()
    gt = gt[mask]
    imputations = imputations[mask]
    pred = imputations.mean(1)
    return np.sqrt(((pred - gt) ** 2).mean()) / std


def compute_pfc(gt, mask, imputations):
    # Compute the proportion of falsely classified entries.
    imputations = np.round(imputations).astype('int')
    gt = gt[mask]
    imputations = imputations[mask]
    categories = sorted(list(set(imputations.ravel()).union(set(gt.ravel()))))
    imputations_cat = [(imputations == category).sum(1)
                       for category in categories]

    if (len(imputations_cat) == 0):
      return 0
    else:
      imputations_cat = np.hstack([x.reshape(-1, 1) for x in imputations_cat])
      pred = np.argmax(imputations_cat, 1)
      return (pred != gt).mean()


# reshape imputation results
results = output_data.reshape(input_data.shape[0], -1, input_data.shape[1])

# define what was imputed
mask = np.isnan(input_data)

# compute NRMSE or PFC for each column
nrmses = []
pfcs = []
for col_id, size in enumerate(one_hot_max_sizes):
    args = groundtruth[:, col_id], mask[:, col_id], results[:, :, col_id]
    if size <= 1:
        nrmse = compute_nrmse(*args)
        if np.isnan(nrmse):
            continue
        nrmses.append(nrmse)
        print('Column %02d, NRMSE: %g' % (col_id + 1, nrmse))
    # else:
    #     pfc = compute_pfc(*args)
    #     pfcs.append(pfc)
    #     print('Column %02d, PFC: %g' % (col_id + 1, pfc))

# print average NRMSE and PFC over all columns
print()
print('NRMSE: %g' % (sum(nrmses) / max(1, len(nrmses))))
# print('PFC: %g' % (sum(pfcs) / max(1, len(pfcs))))
