import os
import os
import sys
import random
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))

def load_train_data(seed = 0):
  # Gather data
  output_path_pos = os.path.join(file_path, './preprocessed/test_data_prep_pos.txt')
  output_path_neg = os.path.join(file_path, './preprocessed/test_data_prep_neg.txt')

  # Load the preprocessed training data
  train_texts = []
  train_labels = []

  with open(output_path_pos) as f:
    train_texts.append(f.read())
    train_labels.append(1)

  with open(output_path_neg) as f:
    train_texts.append(f.read())
    train_labels.append(1)

  random.seed(seed)
  random.shuffle(train_texts)

  return (train_texts, np.array(train_labels))

def load_test_data():
  print("load test data")

