import sys
import os
import numpy as np

sys.path.append('./model')
sys.path.append('./data')
sys.path.append('./embed')

import data
import embed
from sep_cnn_model import SepCNNModel

TOP_K = 20000

def run_training():
  texts, labels = data.load_train_data()

  # create empedding
  input, word_index = embed.sequence_vectorize(texts)
  labels = np.array(labels)

  # create model
  model = RNNModel()

  # pipeline
  num_features = min(len(word_index) + 1, TOP_K)
  model.build(num_features, input_shape=input.shape[1:])

  model.fit(input, labels)
  model.save('saved_models/rnn_model')

# Predict
if __name__ == '__main__':
  run_training()
