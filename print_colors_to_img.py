import csv
import cv2
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

parser.add_argument('--file_folder', action='store', type=str, required=True)

parser.add_argument('--original_file', action='store', type=str, required=True)

parser.add_argument('--imputation_file', action='store', type=str, required=True)

parser.add_argument('--num_imputations', action='store', type=int, required=True)

args = parser.parse_args()

# file_folder = "imputations_vis/bg_0_1_2_imputations/"
# original_file = "data/original_data/forModel_all_LAB.csv"
# imputation_file = "data/imputations/bg_0_1_2_imputations.tsv"

file_folder = args.file_folder
original_file = args.original_file
imputation_file = args.imputation_file

def print_colors(five_reses, original, img_id):
    width = 40
    height = 20
    gap = 20

    color_palettes_to_be_printed = np.append(np.array(original).reshape(1,-1), five_reses, axis=0)
    len = color_palettes_to_be_printed.shape[0]
    h = height * len + gap * (len + 1)
    w = width * color_num

    res_img = np.full((h,w,3), 255)

    height_start = gap

    for ind, group in enumerate(color_palettes_to_be_printed):
        group = group.reshape((1,color_num,3))
        group = cv2.cvtColor(group.astype('uint8'), cv2.COLOR_LAB2BGR)
        # group = cv2.cvtColor(group.astype('uint8'), cv2.COLOR_HSV2BGR)
        for i, c in enumerate(group[0]):
            res_img[height_start:height_start + height, width*i : width * (i + 1),:] = c
        height_start = height_start + height + gap

    cv2.imwrite(file_folder + str(img_id) + ".png", res_img)


color_num = 1 + 18

if __name__ == "__main__":

    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    input_color_columns = [1,2,3]
    for i in range(0, color_num - 1):
        input_color_columns.append(13 + i*9)
        input_color_columns.append(14 + i*9)
        input_color_columns.append(15 + i*9)

    imputation_color_columns = [i - 1 for i in input_color_columns]

    with open(original_file) as f:
        original = csv.reader(f, delimiter=",")
        next(original, None)
        original = list(original)
        f.close()

    with open(imputation_file) as f:
        imputation_results = csv.reader(f, delimiter="\t")
        imputation_results = list(imputation_results)
        index = 0
        for ori in original:
            ori = np.array(ori)
            img_id = int(float(ori[0]))
            ori = ori[input_color_columns]
            ori = [int(float(x) * 255)  for x in ori]

            imputations = np.array([])
            for i in range(index * args.num_imputations, index *args.num_imputations + args.num_imputations):
                res = imputation_results[i]
                res = np.array(res)
                res = res[imputation_color_columns]
                res = [int(float(x) * 255)  for x in res]
                imputations = np.append(imputations, res)
                imputations = imputations.reshape(-1,color_num*3)
            index = index + 1
            print_colors(imputations, ori, img_id)
    f.close()





