from os import makedirs
from os.path import join

import numpy as np
import pandas as pd

feature_file = 'feature_v1'

trained_list = ['0_color_l', '0_color_a', '0_color_b', 
  '1_color_l', '1_color_a', '1_color_b', 
  '2_color_l', '2_color_a', '2_color_b', 
  '3_color_l', '3_color_a', '3_color_b', 
  '4_color_l', '4_color_a', '4_color_b']

mcar_prob = 0.5
random_seed = 239

def save_data(filename, data):
    np.savetxt(filename, data, delimiter='\t')
   
np.random.seed(random_seed)

data = pd.read_csv(join('original_data', feature_file + '.csv'), index_col=0, sep=',')

train_data = data.copy()

for field in trained_list:
  mask = np.random.choice(2, size=data.shape[0], p=[mcar_prob, 1 - mcar_prob])
  col = np.array(train_data[field]).astype('float')
  nw_col = col.copy()
  nw_col[(1 - mask).astype('bool')] = np.nan
  train_data[field] = nw_col

makedirs('train_test_split', exist_ok=True)
save_data(join('train_test_split', '{}_train.tsv'.format(feature_file)), train_data)
save_data(join('train_test_split', '{}_groundtruth.tsv'.format(feature_file)), data)

train_data.to_csv('out.csv')
