#%%

# importring libraries
import os
from argparse import Namespace
from collections import Counter
import json
import re
import string
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split,ShuffleSplit
from tqdm import tqdm_notebook
import nltk.data
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#import sentencepiece as spm
from tqdm import tqdm
import unicodedata
import six
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
#importing module
import logging
from torch.utils.tensorboard import SummaryWriter

#Create and configure logger
logging.basicConfig(filename="checkpoint/BILSTM.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

#Creating an object
logger=logging.getLogger()

#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)
#%%
writer = SummaryWriter('runs/BILSTM_train')
#%%
## Data Loading

def load_directory_data(file):
  """Retrieves the Sentences from the input text file into a Dict,stores and return into Dataframe """
  data = {}
  data["sentence"] = []
  num_lines = sum(1 for line in open(file,'r',encoding='utf-8'))
  with open(file,'r',encoding='utf-8') as f:
    print('\n opened file : '+file)
    for line in tqdm(f, total=num_lines):
        data["sentence"].append(line)
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory,Train):
  """ Specifically for assigning labels and get the train_df or test_df from the pre-defined datasets"""
  if (Train):
    pos_df = load_directory_data(os.path.join(directory, "train_pos.txt"))
    neg_df = load_directory_data(os.path.join(directory, "train_neg.txt"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
  else :
    test_df = load_directory_data(os.path.join(directory, "test.txt"))
    test_df["polarity"] = -1
  return test_df

def loadGloveModel():
  "Loads Glove Embedding as a dictionary"
  print("loading glove model")
  oldpwd=os.getcwd()
  os.chdir("../data/")
  f = open('glove.twitter.27B.200d.txt','r',encoding='utf-8')
  os.chdir(oldpwd)
  gloveModel = {}
  for line in f:
    splitLines = line.split()
    word = splitLines[0]
    wordEmbedding = np.array([float(value) for value in splitLines[1:]])
    gloveModel[word] = wordEmbedding
  print(len(gloveModel)," words loaded!")
  return  gloveModel


# Fetch and Process the dataset files.
def fetch_and_load_datasets():
  """ Initialises both train and test dataframes"""
  oldpwd=os.getcwd()
  os.chdir("../data/cleaned/")
  train_df = load_dataset(os.getcwd(),True)
  test_df = load_dataset(os.getcwd(),False)
  os.chdir(oldpwd)
  GloVe = loadGloveModel()
  return train_df,test_df,GloVe



#%%
# Data Vocabulary class
class SentAnaVocabulary(object):
  """Class to process text and extract vocabulary for mapping ,
       to make it easy to tokenize the words lookup with token or index of the token"""

  def __init__(self, token_to_idx=None, mask_token="<MASK>", add_unk=True, unk_token="<UNK>"):
    """Initialises every dictionary and tokens"""
    if token_to_idx is None:
        token_to_idx = {}
    self._token_to_idx = token_to_idx

    self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    self._add_unk = add_unk
    self._unk_token = unk_token
    self._mask_token = mask_token

    self.mask_index = self.add_token(self._mask_token)
    self.unk_index = -1
    if add_unk:
        self.unk_index = self.add_token(unk_token)

  def add_token(self, token):
    """Update mapping dicts based on the token."""
    if token in self._token_to_idx:
        index = self._token_to_idx[token]
    else:
        index = len(self._token_to_idx)
        self._token_to_idx[token] = index
        self._idx_to_token[index] = token
    return index

  # def get_embedding(self, token):
  #     """Add a list of tokens into the Vocabulary"""
  #     if token in self.Glove:
  #       token_embedding = self.Glove.get(token)
  #     else:
  #       token_embedding = float(0) * 25
  #     return token_embedding

  def lookup_token(self, token):
      """Retrieve the index associated with the token or the UNK index if token isn't present."""
      if self.unk_index >= 0:
          return self._token_to_idx.get(token, self.unk_index)
      else:
          return self._token_to_idx[token]

  def lookup_index(self, index):
      """Return the token associated with the index"""
      if index not in self._idx_to_token:
          raise KeyError("the index (%d) is not in the Vocabulary" % index)
      return self._idx_to_token[index]

  def __len__(self):
      return len(self._token_to_idx)

  def __str__(self):
      return "<Vocabulary(size=%d)>" % len(self)


## Sentence Vectorizer
class SentAnaVectorizer(object):
  """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
  def __init__(self, SA_vocab):
      """
      Assigns an Class Global Object for further vectorizing of data after getting initialised SentAnaVocabulary Object
          SA_vocab (Vocabulary) for mapping words to integers
      """
      self.SA_vocab = SA_vocab
      self.Data_x = torch.tensor([])


  @classmethod
  def from_dataframe(cls, sa_df):
      """Instantiate the vectorizer from the dataset dataframe  """
      SA_vocab = SentAnaVocabulary()
      for index, row in sa_df.iterrows():
        # replace the row.sentence.split() with sentencepiece
          for token in row.sentence.split(' '):
              SA_vocab.add_token(token)
          # self.Data_x[index] = torch.tensor([list(embedding)])
      return cls(SA_vocab)

  def vectorize(self, sentence, vector_length=-1):
    """   Function vectorizes the DataSet using the existing vocab  """

    indices = [self.SA_vocab.lookup_token(token) for token in sentence.split(' ')]
    if vector_length < 0:
        vector_length = len(indices)

    out_vector = np.zeros(vector_length, dtype=np.int64)
    out_vector[:len(indices)] = indices
    # replace it with mask_index or other
    out_vector[len(indices):] = self.SA_vocab.mask_index

    return out_vector


## SentenceDataet
class SentAnaDataset(Dataset):
  def __init__(self, sa_df, vectorizer,max_seq_len):
    """
    Initialises the Dataset and call the vocab and vectorizer
    """
    self.sa_df = sa_df
    self._vectorizer = vectorizer

    measure_len = lambda sentence: len(sentence.split(" "))
    self._max_seq_length = min(max(map(measure_len, sa_df.sentence)),max_seq_len)


  @classmethod
  def load_dataset(cls, sa_df):
    """Load dataset and make a new vectorizer from scratch"""
    return cls(sa_df, SentAnaVectorizer.from_dataframe(sa_df),128)

  def get_vectorizer(self):
    """ returns the vectorizer """
    return self._vectorizer


  def __len__(self):
    return self._target_size

  def get_num_batches(self, batch_size):
    """Given a batch size, return the number of batches in the dataset"""
    return len(self) // batch_size


#%%



class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, weights_matrix , hidden_dim, output_size, batch_size, n_layers, drop_prob=0.5,Device=False):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()
        self.device = Device
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # embedding
        def create_embedding_layer(weights_matrix):
          "Takes Embedding as matrix and load into the embedding layer"
          num_embeddings, embedding_dim = weights_matrix.shape
          emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix),padding_idx=1)
          return emb_layer, num_embeddings, embedding_dim

        emb_layer , vocab_size, embedding_dim = create_embedding_layer(weights_matrix)
        self.embedding = emb_layer

        # LSTM LAyer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,dropout=drop_prob, batch_first=True,bidirectional=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layer
        self.decoder1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim,output_size)
        # self.hidden = self.init_hidden(batch_size)

        self.relu = nn.ReLU()

    def forward(self, x,hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)
        sentence_length = x.size(1) * torch.ones(batch_size)

        embeds = self.embedding(x)


        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, sentence_length, batch_first=True)
        lstm_out, hidden = self.lstm(embeds, hidden)


        # lstm_fw = lstm_out[:, :, :self.hidden_dim]
        # lstm_bw = lstm_out[:, :, self.hidden_dim:]

        #Fetching the hidden state of Backward and Forward
        lstm_out = torch.cat((hidden[0][-2, :, :], hidden[0][-1, :, :]), dim=1)

        lstm_out = self.decoder1(lstm_out)

        lstm_out = self.relu(lstm_out)

        lstm_out = self.dropout(lstm_out)

        lstm_out = self.decoder2(lstm_out)

        return F.log_softmax(lstm_out,dim=1),hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if(self.device):
          hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda())
        else:
          hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_())

        return hidden


#%%

print("Loading dataset")
train_df , test_df ,GloVe= fetch_and_load_datasets()
dataset = SentAnaDataset.load_dataset(train_df)
vectorizer = dataset.get_vectorizer()

print("Creating the embedding matrix")
matrix_len = len(vectorizer.SA_vocab)
weights_matrix = np.zeros((matrix_len, 200))
words_found = 0

for i, word in enumerate(vectorizer.SA_vocab._token_to_idx):
    try:
        weights_matrix[i] = GloVe[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(200,))


train_df, val_df = train_test_split(train_df,shuffle=True,train_size=0.9,test_size=0.1,random_state=42)

train_df.reset_index(drop=True,inplace=True)

val_df.reset_index(drop=True,inplace=True)

outputs = []
for index,row in train_df.iterrows():
  outputs.append(torch.tensor(vectorizer.vectorize(row['sentence'],140)))

val_outputs = []
for index,row in val_df.iterrows():
  val_outputs.append(torch.tensor(vectorizer.vectorize(row['sentence'],140)))

test_outputs = []
for index,row in test_df.iterrows():
  test_outputs.append(torch.tensor(vectorizer.vectorize(row['sentence'],140)))

#%%
print("Created the embedding matrix")
train_x = torch.stack(outputs)
train_y = torch.tensor(train_df['polarity'].values, dtype=torch.long)

val_x = torch.stack(val_outputs)
val_y = torch.tensor(val_df['polarity'].values, dtype=torch.long)

test_x = torch.stack(test_outputs)

train_data=TensorDataset(train_x,train_y)
val_data = TensorDataset(val_x,val_y)
test_data=TensorDataset(test_x)

#TrainData, ValidationData = random_split(train_data,[int(0.9*len(train_data)),len(train_data) - int(0.9*len(train_data))])

pad_idx = vectorizer.SA_vocab.lookup_token('<MASK>')
HIDDEN_DIM = 256
BATCH_SIZE= 64
OUTPUT_DIM = 2
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
EPOCHS =3
batch_size =64
#Device = False

# loss and optimization functions
learningrate=0.0001

cuda_available = True
# Check CUDA
if not torch.cuda.is_available():
    cuda_available = False

Device = torch.device("cuda" if cuda_available else "cpu")


model = SentimentRNN(weights_matrix, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE, N_LAYERS)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE,drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=100)
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
loss_func.to(Device)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
model.train()


#%%
def classDefiner(x):
    if x[0] > x[1]:
        return -1
    return 1

def accuracy(predicted,labels):
    acc = 0
    preds = map(classDefiner,list(predicted.cpu().detach().numpy()))
    labels = list(labels.cpu().detach().numpy())
    preds = list(preds)
    for i in range(0,len(preds)):
      if preds[i] == labels[i]:
        acc+=1
    return acc/len(preds)


#%%
# Model training

epochs = 10
clip=5 # gradient clipping

epoch_bar = tqdm(desc='Epochs',total=epochs,position=0)
train_bar = tqdm(desc='Training',total=len(train_loader),position=1,leave=True)
val_bar = tqdm(desc='Validation',total=len(valid_loader),position=2,leave=True)

# move model to GPU, if available
if(cuda_available):
    model.cuda()

epoch_bar.n=0
# train for some number of epochs
for e in range(epochs):
    epoch_bar.update()
    epoch_bar.set_postfix(lr=learningrate)
    # batch loop
    model.train()
    train_bar.n = 0
    val_bar.n = 0
    running_loss = 0
    train_accuracy = 0
    hidden = model.init_hidden(BATCH_SIZE)
    index = 0
    for inputs, labels in train_loader:
        train_bar.update()
        if(cuda_available):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        hidden = tuple([each.data for each in hidden])

        # --------------------------------------
        # step 1. zero the gradients
        model.zero_grad()

        # step 2. compute the output
        predictions,hidden  = model(inputs,hidden)

        #print(predictions)
        loss = loss_func(predictions, labels)
        loss.backward(retain_graph=True)
        running_loss += (loss.detach()  - running_loss) / (index + 1)
        train_accuracy += accuracy(predictions,labels) / (index + 1)
        index+=1
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        train_bar.set_postfix(loss=running_loss,acc=train_accuracy, epoch=e)
        # loss stats

    valid_hidden = model.init_hidden(BATCH_SIZE)
    val_losses = []
    model.eval()
    running_val_loss = 0
    val_accuracy = 0
    index = 0
    for inputs, labels in valid_loader:
        val_bar.update()
        if(cuda_available):
            inputs, labels = inputs.cuda(), labels.cuda()

        valid_hidden = tuple([each.data for each in valid_hidden])

        output,valid_hidden = model(inputs,valid_hidden)

        val_loss = loss_func(output, labels)
        running_val_loss += (val_loss.detach()  - running_val_loss) / (index + 1)
        val_accuracy += accuracy(predictions,labels) / (index + 1)
        index+=1
        val_losses.append(val_loss.item())
        val_bar.set_postfix(loss=val_loss,acc=val_accuracy,epoch=e)

    # Optimizer Learning Rate
    learningrate = optimizer.param_groups[0]['lr']

    #LearningRate Scheduler
    my_lr_scheduler.step()

    writer.add_scalar('Loss/train', running_loss, e)
    writer.add_scalar('Loss/val', running_val_loss, e)
    writer.add_scalar('Accuracy/train', train_accuracy, e)
    writer.add_scalar('Accuracy/val', val_accuracy, e)

    logger.info('train_loss == '+str(running_loss)+'at epoch'+str(e+1))
    logger.info('val_loss == '+str(running_val_loss)+'at epoch'+str(e+1))
    logger.info('train_acc == '+str(train_accuracy)+'at epoch'+str(e+1))
    logger.info('val_acc == '+str(val_accuracy)+'at epoch'+str(e+1))
    torch.save({
    'epoch': e+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': running_loss,
    'val_loss': running_val_loss,
    'train_acc': train_accuracy,
    'val_acc': val_accuracy,
    }, 'checkpoint/entire_model_BiLSTM_'+str(e+1)+'.pt')


valid_hidden = model.init_hidden(BATCH_SIZE)
model.eval()
val_op = []
for inputs, labels in valid_loader:
    if(cuda_available):
        inputs, labels = inputs.cuda(), labels.cuda()

    valid_hidden = tuple([each.data for each in valid_hidden])

    val_predictions,valid_hidden = model(inputs,valid_hidden)

    op3 = val_predictions.cpu()
    val_preds = map(classDefiner,list(op3.detach().numpy()))
    for item in list(val_preds):
      val_op.append(item)

print(val_df.shape)
print(len(val_op))

val_df['predictions'] = np.zeros(val_df.shape[0])

for i in range(0,len(val_op)):
  val_df.loc[i,'predictions'] = val_op[i]

val_df.to_csv('checkpoint/Misclassification.csv')

model.eval()
test_hidden = model.init_hidden(100)
print(type(test_loader))

id=1

for inputs in test_loader:
    if(cuda_available):
        inputs = inputs[0].cuda()

    test_hidden = tuple([each.data for each in test_hidden])

    test_predictions,test_hidden = model(inputs,test_hidden)
    with open('checkpoint/output_BiLSTM_new.txt','a+',encoding ="utf-8") as fp:
        op2 = test_predictions.cpu()
        preds = map(classDefiner,list(op2.detach().numpy()))
        for item in list(preds):
            fp.write("{},{}\n".format(id,item))
            id+=1

PATH = "checkpoint/entire_model_BiLSTM_New.pt"

# Save
torch.save(model, PATH)

# %%
