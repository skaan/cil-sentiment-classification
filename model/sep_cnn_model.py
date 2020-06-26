from base_model import BaseModel
import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D

checkpoint_path = "checkpoints/rnn/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

TOP_K = 20000

class SepCNNModel(BaseModel):

  def build(
      self,
      num_features,
      input_shape,
      blocks=2,
      dropout_rate=0.1,
      embedding_dim=200,
      filters=64,
      kernel_size=3,
      pool_size=3):

    # create model
    self.model = models.Sequential()
    self.model.add(Embedding(
        input_dim=num_features,
        output_dim=embedding_dim,
        input_length=input_shape[0]))

    for _ in range(blocks-1):
      model.add(Dropout(rate=dropout_rate))
      model.add(SeparableConv1D(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same'))
      model.add(SeparableConv1D(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same'))
      model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(2, activation='sigmoid'))

    # compile model with loss and optimizer
    self.model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(1e-4),
      metrics=['acc']
    )


  def fit(self, input, labels, epochs=50):
    self.model.fit(
      input,
      labels,
      epochs=epochs,
      validation_split=0.1,
      verbose=2, 
    )


  def predict(self, prediction_data):
    predictions = self.model.predict(prediction_data)
    return predictions


  def save(self, path):
    self.model.save(path) 
