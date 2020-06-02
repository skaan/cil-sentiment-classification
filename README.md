# Prerequisites

### Git LFS
Install git LFS to push/pull train data, weights, big files. Follow this link for installing:
https://github.com/git-lfs/git-lfs/wiki/Installation

Git LFS for Ubuntu & Debian. Make sure you run ```git lfs install``` within the git repository.
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```
Pull works normally.

If you want to push big files (>100MB), track them first, and then add them:
```
git lfs track FILEPATH
git add FILEPATH
```
After tracking, don't forget to add ```.gitattributes```.

### Preprocessing
Install the following to use hashtag split and spelling correction in preprocessing:
```
pip install pyenchant
pip install wordsegment
pip install textblob
```
Text Blob needs nltk and nltk.download('punkt')

#### Installing the Enchant C library for `pyenchant'

See [docs](https://pyenchant.github.io/pyenchant/install.html#installing-the-enchant-c-library)

### Models
Install the following if you want to use BERT:
```
pip install bert-for-tf2
```

# Resources
## State of the art (general)
* List SOTAs for open source DS: http://nlpprogress.com/english/sentiment_analysis.html
* SOTA for many DS: https://arxiv.org/pdf/1906.08237.pdf
* SOTAs evaluated on different DS: https://www.aclweb.org/anthology/W17-5202.pdf

## Sentiment Analysis specific paper: 
* BERT:
    * How to Fine-Tune BERT for Text Classification
      https://arxiv.org/pdf/1905.05583.pdf
    * BERT as YouTube video:
      https://www.youtube.com/watch?v=OR0wfP2FD3c
    * Bert Trained for Twitter Sentiment analysis:
      https://github.com/XiaoQQin/BERT-fine-tuning-for-twitter-sentiment-analysis
    * Sentiment analysis using BERT (pre-training language representations):
      https://iust-deep-learning.github.io/972/static_files/project_reports/sent.pdf
* Really amazing accomplishments towards Sentiment Analysis:(Article, Github)
https://paperswithcode.com/task/sentiment-analysis

## Twitter specific methods
* Lexicon based sentiment analysis for social networks: https://www.researchgate.net/publication/283318830_Lexicon-Based_Sentiment_Analysis_in_the_Social_Web?ev=prf_pub
* Lexicon based for twitter: https://www.researchgate.net/publication/270794878_Sentiment_Analysis_Twitter_dengan_Kombinasi_Lexicon_Based_dan_Double_Propagation
* Competition on twitter data. Subtask E was to give words and sentences polarity scores. Would still need to google what the individual teams did, but found it already for some. https://docs.google.com/document/d/1WV-XTvQDpuH_IfKrjzeZ361s1ykcskDNNuOV3oI39_c/edit
* Spelling corrector: https://www.researchgate.net/publication/284344945_Context-Aware_Spelling_Corrector_for_Sentiment_Analysis?ev=prf_pub
* Detection and scoring of Internet slang: https://www.researchgate.net/publication/283318703_Detection_and_Scoring_of_Internet_Slangs_for_Sentiment_Analysis_Using_SentiWordNet?ev=prf_pub

## Overview over general techniques
* General short overview: https://www.kdnuggets.com/2018/03/5-things-sentiment-analysis-classification.html
* Overview over 54 papers and what they've used: https://www.sciencedirect.com/science/article/pii/S2090447914000550

## "Creative" things
Things where we can bring in own ideas.

### Preprocessing
* Break up hash tags
* Score smileys. e.g. :D :p . We could use this: https://en.wikipedia.org/wiki/List_of_emoticons and e.g. just replace them by the wikipedia word.
* Combine lexicon based with classic ML. E.g. get rotten tomato score for movie titles
* Twitter specific things: Correct misspelled words, deal with slang words/grammar somehow (I don't know if standard stemmers can stem them for example)

### Pretraining
* BERT NSP: Predict if part of same tweet instead of predict if subsequent sentence. I'm actually not sure if we should do that at all because I think it especially benefits Q&A prediction. But could do pretraining on other twitter data like that.


## Similar data sets
* List of sentiment data sets: https://blog.cambridgespark.com/50-free-machine-learning-datasets-sentiment-analysis-b9388f79c124
* Polarity (negative 0 - positve 1) of words/smileys/hashtags used on twitter: http://alt.qcri.org/semeval2015/task10/data/uploads/semeval2015-task10-subtaske-testdata-gold.txt
* https://www.trackmyhashtag.com/twitter-dataset
* 467 Million tweets: http://snap.stanford.edu/data/bigdata/twitter7/
