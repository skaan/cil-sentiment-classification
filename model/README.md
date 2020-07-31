## Models

All models implement interface and can be called etc. and implement the following methods:

 - train
 - predict
 - pretrain: if not included in every model then say where this is included. "The training procedure as a whole is included in ... . This includes Pretraining, training, prediction and logging of val_acc".
 - load weights
 - ...

The models implemented include the following:

 - ```SepCNN```: Very brief description, maybe just link to paper
 - ```BiLSTM with self attention``: Very brief description, maybe just link to paper
 - ```Stacked BiLSTM``: Very brief description, maybe just link to paper
 - ```Stacked GRU``: Very brief description, maybe just link to paper







## TODO: change that into requirements.txt 
Please install the following to use hashtag split and spelling correction in preprocessing.

Make sure the Model is executed from model folder and the Bert Config file are also in the same folder.
the File input is train_pos_full_prep.txt,train_neg_full_prep.txt,test_data.txt
```
pip install bert-for-tf2
pip install tensorflow
(tensorflow must be verion 2)
pip install tqdm
pip install pandas
pip intall numpy
```
