import os
import sys

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, './preprocessing'))
from pipeline_1 import Pipeline


# Setup file paths
input_path_pos = os.path.join(file_path, './data/raw/part_train_pos.txt')
input_path_neg = os.path.join(file_path, './data/raw/part_train_pos.txt')
output_path_pos = os.path.join(file_path, './data/preprocessed/test_data_prep_pos.txt')
output_path_neg = os.path.join(file_path, './data/preprocessed/test_data_prep_neg.txt')


# Preprocess
preprocessing = Pipeline()
preprocessing.process(
  [ input_path_pos, input_path_neg ], 
  [ output_path_pos, output_path_neg ]
)