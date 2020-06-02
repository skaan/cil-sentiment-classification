from base_model import BaseModel
import tensorflow as tf
import os
import numpy as np

checkpoint_path = "checkpoints/weights_rnn.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

TOP_K = 20000

class RNNModel(BaseModel):

  def __init__(self, weights_path=None):
    super().__init__()

  def build(self, word_index):
    num_features = min(len(word_index) + 1, TOP_K)

    self.model = tf.keras.Sequential([
      tf.keras.layers.Embedding(num_features, 64),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
    ])

    self.model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(1e-4),
      metrics=['accuracy']
    )

  def fit(self, input, labels, epochs=50):
    print("Started training")
    # callbacks = [
    #   tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
    # ]

    self.model.fit(
      input,
      labels,
      epochs=epochs,
      #callbacks=callbacks,
      validation_split=0.1,
      verbose=2, 
    )

  def predict(self, prediction_data):
    predictions = self.model.predict(prediction_data)
    return predictions

  def save(self, path):
    self.model.save(path) 
