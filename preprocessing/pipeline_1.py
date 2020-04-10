'''
Pipeline:
1. Break up hash tags
2. Replace emoticons
3. Appos remove
4. Remove stopwords
 # 5. Spelling correction
6. Wordnet lemmatize
'''

import os

from hashtag_split_ww import HashtagSplitWW
from emoticon_replace import EmoticonReplace
from appos_remove import ApposRemove
from stopwords_remove import StopwordsRemove
#from spelling_correction import SpellingCorrection
from wordnet_lemma import WordNetLemma


# Init steps
hs = HashtagSplitWW()
er = EmoticonReplace()
ar = ApposRemove()
#sc = SpellingCorrection()
sr = StopwordsRemove()
wl = WordNetLemma()

# execute pipeline
file_path = os.path.dirname(__file__)
data = 'pos'

for i in range(2):

    # data paths
    path_0 = os.path.join(file_path, "../data/part_train_" + data + ".txt")
    path_1 = os.path.join(file_path, "../data/train_" + data + "_full_1.txt")
    path_2 = os.path.join(file_path, "../data/train_" + data + "_full_2.txt")
    path_3 = os.path.join(file_path, "../data/train_" + data + "_full_3.txt")
    path_4 = os.path.join(file_path, "../data/train_" + data + "_full_4.txt")
    path_5 = os.path.join(file_path, "../data/train_" + data + "_full_prep.txt")

    # set paths
    hs.set_paths(path_0, path_1)
    er.set_paths(path_1, path_2)
    ar.set_paths(path_2, path_3)
    sr.set_paths(path_3, path_4)
    #sc.set_paths(path_4, path_5)
    wl.set_paths(path_4, path_5)

    # run
    hs.run()
    er.run()
    ar.run()
    sr.run()
    #sc.run()
    wl.run()

    data = 'neg'
