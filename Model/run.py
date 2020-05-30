from rnn_model import RNNModel

# TODO: Load datasets from path
import tensorflow_datasets as tfds

# Get datasets
train_data, test_data = tfds.load(
    'imdb_reviews/subwords8k', 
    split=['train[:10%]', 'test[:10%]'],
    as_supervised=True)

# Prepare dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = (
  train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(BATCH_SIZE)
)

test_dataset = (
  test_data
    .padded_batch(BATCH_SIZE)
)

model = RNNModel()

# pipeline
model.fit(train_dataset)
model.predict(test_dataset)