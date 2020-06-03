import sys
import os
import numpy as np

sys.path.append('./model')
sys.path.append('./data')
sys.path.append('./embed')

import data
import embed
from rnn_model import RNNModel

texts, labels = data.load_train_data()

# create empedding
input, word_index = embed.sequence_vectorize(texts)
labels = np.array(labels)

# create model
model = RNNModel()

# pipeline
model.build(word_index)

model.fit(input, labels)
model.save('saved_models/rnn_model')
