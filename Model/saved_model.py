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

  def __init__(self, model_path):
    super().__init__()
    self.model = tf.keras.models.load_model(model_path)

  def fit(self, training_data):
    self.model.fit(
      training_data, 
      epochs=10,
      callbacks=[cp_callback]
    )

  def evaluate(self, prediction_data):
    predictions = self.model.predict(prediction_data)
    return predictions
