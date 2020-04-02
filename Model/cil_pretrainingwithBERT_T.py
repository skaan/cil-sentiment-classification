#%%
import tensorflow as tf
import tensorflow_hub as hub
print(tf.__version__)
import numpy as np
import pandas as pd
import re
import os
import bert
from  tqdm import tqdm
from bert.tokenization.bert_tokenization import FullTokenizer
import nltk.data
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#%%

data_df = pd.read_csv(r'C:/Users/potta/Documents/CIL/training.1600000.processed.noemoticon.csv',sep=',',encoding='ISO-8859-1',header = None)

data_df.head(10)

data_df[0].value_counts()

Train_df = data_df.drop(columns=[2,3,4])

bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)

#%%


MAX_SEQ_LEN=128
input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,name="segment_ids")


#%%

def get_masks(tokens, max_seq_length):
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))
 
def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

#%%

FullTokenizer=bert.bert_tokenization.FullTokenizer
 
vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()
 
do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()
 
tokenizer=FullTokenizer(vocab_file,do_lower_case)
 
def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens,)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

	
def create_single_input(sentence,MAX_LEN):
  
  stokens = tokenizer.tokenize(sentence)
  
  stokens = stokens[:MAX_LEN]
  
  stokens = ["[CLS]"] + stokens + ["[SEP]"]
 
  ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)
  masks = get_masks(stokens, MAX_SEQ_LEN)
  segments = get_segments(stokens, MAX_SEQ_LEN)
 
  return ids,masks,segments
 
def create_input_array(sentences):
 
  input_ids, input_masks, input_segments = [], [], []
 
  for sentence in tqdm(sentences,position=0, leave=True):
  
    ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2)
 
    input_ids.append(ids)
    input_masks.append(masks)
    input_segments.append(segments)
 
  return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

#%%


def clean_tweets(tweet):
    stop_words = set(stopwords.words('english'))
    #after tweet preprocessing the colon symbol left remain after #removing mentions
    tweet = re.sub(r'‚Ä¶', '', tweet)
    #after tweet 
    tweet = re.sub(r'@([^\s]+)', '', tweet)
    #replace the https
    tweet = re.sub(r'http[s]?[:]?[\s]?\/\/(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+','',tweet)

    word_tokens = word_tokenize(tweet)
    #filter using NLTK library append it to a string
    filtered_tweet = [w for w in word_tokens if not w in stop_words]
    filtered_tweet = []
    #looping through conditions
    for w in word_tokens:
    #check tokens against stop words and punctuations
        if w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)


#%%
train_sentences = [clean_tweets(x) for x in Train_df.ix[:,5].values]
train_y = Train_df[0].values

#%%

x = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(6, activation="sigmoid", name="dense_output")(x)
 
model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
 
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

inputs=create_input_array(train_sentences)
 
model.fit(inputs,train_y,epochs=1,batch_size=32,validation_split=0.2,shuffle=True)

#%%

test_df=pd.read_csv(r'C:/Users/potta/Documents/CIL/testdata.manual.2009.06.14.csv',sep=',',encoding='ISO-8859-1',header = None)
 
test_sentences = [clean_tweets(x) for x in test_df.ix[:,5].values]
 
test_inputs=create_input_array(test_sentences)
 
test_predicted = model.predict(test_inputs)

# %%
