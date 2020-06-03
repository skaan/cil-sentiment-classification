import os
import os
import sys
import random

file_path = os.path.dirname(os.path.abspath(__file__))
get_prep_path = lambda file: os.path.join(file_path, f"./preprocessed/{file}")

def load_train_data(is_full=True, seed = 0):
  # Gather data
  full_train_pos_path = get_prep_path("train_pos_full.txt")
  full_train_neg_path = get_prep_path("train_neg_full.txt")
  part_train_pos_path = get_prep_path("part_train_pos.txt")
  part_train_neg_path = get_prep_path("part_train_neg.txt")

  # Load the preprocessed training data
  features = []
  labels = []

  if is_full:
    pos_path = full_train_pos_path
    neg_path = full_train_neg_path
  else:
    pos_path = part_train_pos_path
    neg_path = part_train_neg_path

  with open(pos_path) as f:
    for line in f:
      line = line.strip()
      features.append(line)
      labels.append(1)

  with open(neg_path) as f:
    for line in f:
      line = line.strip()
      features.append(line)
      labels.append(0)

  random.seed(seed)
  random.shuffle(features)
  random.shuffle(labels)

  return features, labels


def load_test_data():
  test_path = get_prep_path("test_prep.txt")

  # Load the preprocessed test data
  features = []

  with open(test_path) as f:
    for line in f:
      line = line.strip()
      features.append(line)

  return features
