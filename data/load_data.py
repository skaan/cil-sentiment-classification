import os
import os
import sys
import random

file_path = os.path.dirname(os.path.abspath(__file__))
get_prep_path = lambda file: os.path.join(file_path, f"./mst/{file}")

def load_train_data(seed = 0):
  # Gather data
  pos_path = get_prep_path("train_pos.txt")
  neg_path = get_prep_path("train_neg.txt")

  # Load the preprocessed training data
  features = []
  labels = []

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
  test_path = get_prep_path("test.txt")

  # Load the preprocessed test data
  features = []

  with open(test_path) as f:
    for line in f:
      line = line.strip()
      features.append(line)

  return features
