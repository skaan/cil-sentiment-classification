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
  features = []
  labels = []

  with open(output_path_pos) as f:
    for line in f:
      line = line.strip()
      features.append(line)
      labels.append(1)

  with open(output_path_neg) as f:
    for line in f:
      line = line.strip()
      features.append(line)
      labels.append(0)

  random.seed(seed)
  random.shuffle(features)
  random.shuffle(labels)

  return features, labels

def load_test_data():
  print("load test data")