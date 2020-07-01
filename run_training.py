import sys
import os
import numpy as np

sys.path.append('./model')
sys.path.append('./data')
sys.path.append('./embed')

import data
import embed

#from rnn_model import RNNModel
from sep_cnn_model import SepCNNModel


TOP_K = 20000

def run_training():
  texts, labels = data.load_train_data()

  # create empedding
  input, word_index = embed.sequence_vectorize(texts)
  labels = np.array(labels)

  # create model
  model = SepCNNModel()

  # pipeline
  num_features = min(len(word_index) + 1, TOP_K)
  model.build(num_features)

  model.fit(input[:1000], labels[:10])
  model.save('saved_models/rnn_model')

# Predict
if __name__ == '__main__':
  run_training()
