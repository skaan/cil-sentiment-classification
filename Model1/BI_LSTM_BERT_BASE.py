#%%
import os
import math
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import re

#!pip install bert-for-tf2
import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

def load_directory_data(file):
  data = {}
  data["sentence"] = []
  num_lines = sum(1 for line in open(file,'r'))
  with open(file,'r') as f:
    print('\n opened file : '+file)
    for line in tqdm(f, total=num_lines):
        data["sentence"].append(line)
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory,Train):
  if (Train):
    pos_df = load_directory_data(os.path.join(directory, "train_pos_full_prep.txt"))
    neg_df = load_directory_data(os.path.join(directory, "train_neg_full_prep.txt"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
  else :
    test_df = load_directory_data(os.path.join(directory, "test_data.txt"))
    test_df["polarity"] = -1
  return test_df

# Download and process the dataset files.
def fetch_and_load_datasets():
  # dataset = tf.keras.utils.get_file(fname="aclImdb.tar.gz",origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
  #     extract=True)
  oldpwd=os.getcwd()
  os.chdir("../data/")
  train_df = load_dataset(os.getcwd(),True)
  test_df = load_dataset(os.getcwd(),False)
  os.chdir(oldpwd)
  return train_df,test_df


#%%
class Vocabtokenizer:
    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "polarity"

    def __init__(self, tokenizer: FullTokenizer, sample_size=None, max_seq_len=128):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        train, test = fetch_and_load_datasets()
        # Arranging the Sentence in the length-wise and reindexing
        train, test = map(lambda df: df.reindex(df[Vocabtokenizer.DATA_COLUMN].str.len().sort_values().index),[train, test])
                
        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(int(sample_size/10))
            # train, test = map(lambda df: df.sample(sample_size), [train, test])
        
        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad,[self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[Vocabtokenizer.DATA_COLUMN], row[Vocabtokenizer.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)


# %%

# BERT_BASE uncased model
bert_model_name="uncased_L-12_H-768_A-12"
bert_ckpt_dir    = os.path.join(bert_model_name)
bert_ckpt_file   = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
data = Vocabtokenizer(tokenizer,sample_size=781*128,max_seq_len=128)

#%%
print("train_x", data.train_x.shape)
print("train_x_token_types", data.train_x_token_types.shape)
print("train_y", data.train_y.shape)

print("test_x", data.test_x.shape)

print("max_seq_len", data.max_seq_len)

def create_learning_rate_scheduler(max_learn_rate=5e-5,end_learn_rate=1e-7,warmup_epoch_count=10,total_epoch_count=90):
    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

def create_model(max_seq_len):
  """Creates a classification model."""

  bert_params = bert.params_from_pretrained_ckpt(bert_ckpt_dir)
  l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
  #token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
  #output         = bert([input_ids, token_type_ids])
  output         = l_bert(input_ids)

  forward_layer = keras.layers.LSTM(10, return_sequences=True)
  backward_layer = keras.layers.LSTM(10, activation='relu', return_sequences=True, go_backwards=True)
  bicell = keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(output)
  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bicell)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=2, activation="softmax")(logits)

  
  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  # load the pre-trained model weights
  # load_stock_weights(l_bert, bert_ckpt_file)
  bert.load_bert_weights(l_bert, bert_ckpt_file) 

  model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

  model.summary()
        
  return model

#%%

model = create_model(data.max_seq_len)
#%%
log_dir = ".\\logs\\"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

total_epoch_count = 20
model.fit(x=data.train_x, y=data.train_y,
          validation_split=0.1,
          batch_size=16,
          shuffle=True,
          epochs=total_epoch_count,
          verbose = 2,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=20,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                     tensorboard_callback])


model.save_weights('./BILSTM_BERT_weights.h5', overwrite=True)

# %%
train_loss, train_acc = model.evaluate(data.train_x, data.train_y)
test_y = model.predict(data.test_x,batch_size=8,verbose=1)

np.savetxt('testprediction.csv', test_y ,delimiter=',')

print("train acc", train_acc)

model.save('my_model.h5')


pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!"
]

tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
pred_tokens    = map(tokenizer.tokenize, pred_sentences)
pred_tokens    = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

print('pred_token_ids', pred_token_ids.shape)

res = model.predict(pred_token_ids).argmax(axis=-1)

for text, sentiment in zip(pred_sentences, res):
  print(" text:", text)
  print("  res:", ["negative","positive"][sentiment])