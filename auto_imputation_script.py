import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

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

parser.add_argument('--field', action='store', type=str,
                    nargs='+', required=True)

args = parser.parse_args()

field = args.field

data = pd.read_csv('data/original_data/' + 'forModel_all_LAB_test.csv', index_col=0, sep=',')
data_train = pd.read_csv('data/original_data/' + 'forModel_all_LAB_train.csv', index_col=0, sep=',')

ONE_HOT_MAX_SIZES = '1 1 1 11 1 13 0 118 7 1 1 1 1 1 1 2 45 7 1 1 1 1 1 1 4 39 7 1 1 1 1 1 1 6 37 7 1 1 1 1 1 1 8 53 7 1 1 1 1 1 1 10 39 7 1 1 1 1 1 1 12 39 7 1 1 1 1 1 1 14 43 7 1 1 1 1 1 1 16 31 7 1 1 1 1 1 1 18 45 7 1 1 1 1 1 1 20 71 7 1 1 1 1 1 1 22 41 7 1 1 1 1 1 1 24 39 7 1 1 1 1 1 1 26 47 7 1 1 1 1 1 1 28 53 7 1 1 1 1 1 1 30 41 7 1 1 1 1 1 1 32 63 7 1 1 1 1 1 1 34 57 7 1 1 1 1 1 1 '
MODEL = ''
TEST_NUM = 200
IMPUTATION_NUM = 10

## Name
field_name = field.copy()
name_list = ''
if 'bg' in field_name:
  test_name = 'bg_'
  field_name.remove('bg')
else: 
  test_name = ''

name_list = [int(i) for i in field_name]
name_list.sort()

for each in name_list:
  test_name +=  str(each) + '_'

data_test = data.copy()

for each in field:
  three_color = [each+'_color_l', each+'_color_a', each+'_color_b']
  if each == 'bg':
    three_color = ['img_bg_color_l', 'img_bg_color_a', 'img_bg_color_b']
  for one in three_color:
    mask = np.random.choice(2, size=data.shape[0], p=[1, 0])
    col = np.array(data_test[one]).astype('float')
    nw_col = col.copy()
    nw_col[(1 - mask).astype('bool')] = np.nan
    data_test[one] = nw_col

df_row = pd.concat([data_test, data_train])

np.savetxt('data/train_test_split/' + '{}to_be_imputed.tsv'.format(test_name), df_row, delimiter='\t')

os.system('python vis_impute.py --num_imputations {} --input_file data/train_test_split/{}to_be_imputed.tsv --output_file data/imputations/{}imputations.tsv --one_hot_max_sizes 1 1 1 11 1 13  0 118 7 1 1 1 1 1 1 2 45 7 1 1 1 1 1 1 4 39 7 1 1 1 1 1 1 6 37 7 1 1 1 1 1 1 8 53 7 1 1 1 1 1 1 10 39 7 1 1 1 1 1 1 12 39 7 1 1 1 1 1 1 14 43 7 1 1 1 1 1 1 16 31 7 1 1 1 1 1 1 18 45 7 1 1 1 1 1 1 20 71 7 1 1 1 1 1 1 22 41 7 1 1 1 1 1 1 24 39 7 1 1 1 1 1 1 26 47 7 1 1 1 1 1 1 28 53 7 1 1 1 1 1 1 30 41 7 1 1 1 1 1 1 32 63 7 1 1 1 1 1 1 34 57 7 1 1 1 1 1 1'.format(IMPUTATION_NUM, test_name, test_name))

full_imputations = np.loadtxt('data/imputations/{}imputations.tsv'.format(test_name), delimiter='\t')

part_imputations = full_imputations[0:TEST_NUM * IMPUTATION_NUM]
np.savetxt('data/imputations/part_{}imputations.tsv'.format(test_name), part_imputations, delimiter='\t')

os.system('python print_colors_to_img.py --num_imputations {} --file_folder imputations_vis/{}imputations/ --original_file data/original_data/forModel_all_LAB_test.csv --imputation_file data/imputations/part_{}imputations.tsv'.format(IMPUTATION_NUM, test_name, test_name))
