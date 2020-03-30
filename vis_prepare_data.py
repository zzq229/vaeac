from os import makedirs
from os.path import join
from argparse import ArgumentParser

import numpy as np
import pandas as pd

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

parser.add_argument('--input_name', type=str, action='store', required=True,
                    help='Path to the input .tsv file. ' +
                         'NaNs are missing values.')

parser.add_argument('--prob', type=float, action='store', required=True, 
                    choices=[ArgParseRange(0, 1)], 
                    help='Probability to make what percent data be NaNs')

parser.add_argument('--seed', type=int, action='store', required=True,
                    help='Random seed')


args = parser.parse_args()

feature_file = args.input_name

mask_list = ['img_bg_color_l', 'img_bg_color_a', 'img_bg_color_b', 
  '0_color_l', '0_color_a', '0_color_b',
  '1_color_l', '1_color_a', '1_color_b', 
  '2_color_l', '2_color_a', '2_color_b', 
  '3_color_l', '3_color_a', '3_color_b', 
  '4_color_l', '4_color_a', '4_color_b',
  '5_color_l', '5_color_a', '5_color_b',
  '6_color_l', '6_color_a', '6_color_b',
  '7_color_l', '7_color_a', '7_color_b',
  '8_color_l', '8_color_a', '8_color_b',
  '9_color_l', '9_color_a', '9_color_b',
  '10_color_l', '10_color_a', '10_color_b',
  '11_color_l', '11_color_a', '11_color_b',
  '12_color_l', '12_color_a', '12_color_b',
  '13_color_l', '13_color_a', '13_color_b',
  '14_color_l', '14_color_a', '14_color_b',
  '15_color_l', '15_color_a', '15_color_b',
  '16_color_l', '16_color_a', '16_color_b',
  '17_color_l', '17_color_a', '17_color_b']

mcar_prob = args.prob
random_seed = args.seed

def save_data(filename, data):
    np.savetxt(filename, data, delimiter='\t')
   
np.random.seed(random_seed)

data = pd.read_csv(join('data/original_data', feature_file + '.csv'), index_col=0, sep=',')

train_data = data.copy()

for field in mask_list:
  mask = np.random.choice(2, size=data.shape[0], p=[mcar_prob, 1 - mcar_prob])
  col = np.array(train_data[field]).astype('float')
  nw_col = col.copy()
  nw_col[(1 - mask).astype('bool')] = np.nan
  train_data[field] = nw_col


makedirs('data/train_test_split', exist_ok=True)
save_data(join('data/train_test_split', '{}_train.tsv'.format(feature_file)), train_data)
save_data(join('data/train_test_split', '{}_groundtruth.tsv'.format(feature_file)), data)
