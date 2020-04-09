import os
import numpy as np
import pandas as pd

os.system('python vis_prepare_data.py --input_name forModel_all_LAB --prob 0.5 --seed 31415')

os.system('python vis_impute.py --input_file data/train_test_split/forModel_all_LAB_train.tsv --output_file data/imputations/forModel_all_LAB_imputations.tsv --one_hot_max_sizes 1 1 1 11 1 13  0 118 7 1 1 1 1 1 1 2 45 7 1 1 1 1 1 1 4 39 7 1 1 1 1 1 1 6 37 7 1 1 1 1 1 1 8 53 7 1 1 1 1 1 1 10 39 7 1 1 1 1 1 1 12 39 7 1 1 1 1 1 1 14 43 7 1 1 1 1 1 1 16 31 7 1 1 1 1 1 1 18 45 7 1 1 1 1 1 1 20 71 7 1 1 1 1 1 1 22 41 7 1 1 1 1 1 1 24 39 7 1 1 1 1 1 1 26 47 7 1 1 1 1 1 1 28 53 7 1 1 1 1 1 1 30 41 7 1 1 1 1 1 1 32 63 7 1 1 1 1 1 1 34 57 7 1 1 1 1 1 1')

os.system('python vis_evaluate_results.py --input_file data/train_test_split/forModel_all_LAB_train.tsv --imputed_file data/imputations/forModel_all_LAB_imputations.tsv --groundtruth data/train_test_split/forModel_all_LAB_groundtruth.tsv --one_hot_max_sizes 1 1 1 11 1 13  0 118 7 1 1 1 1 1 1 2 45 7 1 1 1 1 1 1 4 39 7 1 1 1 1 1 1 6 37 7 1 1 1 1 1 1 8 53 7 1 1 1 1 1 1 10 39 7 1 1 1 1 1 1 12 39 7 1 1 1 1 1 1 14 43 7 1 1 1 1 1 1 16 31 7 1 1 1 1 1 1 18 45 7 1 1 1 1 1 1 20 71 7 1 1 1 1 1 1 22 41 7 1 1 1 1 1 1 24 39 7 1 1 1 1 1 1 26 47 7 1 1 1 1 1 1 28 53 7 1 1 1 1 1 1 30 41 7 1 1 1 1 1 1 32 63 7 1 1 1 1 1 1 34 57 7 1 1 1 1 1 1')
