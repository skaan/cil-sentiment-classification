from base_model import BaseModel
import tensorflow as tf
import os

checkpoint_path = "checkpoints/weights_rnn.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

class RNNModel(BaseModel):

  def __init__(self, weights_path=None):
    super().__init__()

    # TODO: pass down best vocabulary size from embedding step
    self.model = tf.keras.Sequential([
      tf.keras.layers.Embedding(8185, 64),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
    ])

    self.model.compile(
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.Adam(1e-4),
      metrics=['accuracy']
    )

    if weights_path is not None:
      self.model.load_weights(weights_path)

  def fit(self, training_data):
    self.model.fit(
      training_data, 
      epochs=10,
      callbacks=[cp_callback]
    )
    self.model.save('saved_models/rnn_model') 

  def predict(self, prediction_data):
    predictions = self.model.predict(prediction_data)
    return predictions
