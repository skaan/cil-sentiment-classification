'''
Pipeline:
1. Break up hash tags
2. Replace emoticons
3. Appos remove
4. Spelling correction
5. Remove stopwords
6. Wordnet lemmatize
'''

from hashtag_split_ww import HashtagSplitWW
from emoticon_replace import EmoticonReplace
from appos_remove import ApposRemove
from spelling_correction import SpellingCorrection
from stopwords_remove import StopwordsRemove
from wordnet_lemma import WordNetLemma


# Init steps
hs = HashtagSplitWW()
er = EmoticonReplace()
ar = ApposRemove()
sc = SpellingCorrection()
sr = StopwordsRemove()
wl = WordNetLemma()

# execute pipeline
data = 'pos'

for i in range(2):

    # data paths
    path_0 = "../data/train_" + data + "_full.txt"
    path_1 = "../data/train_" + data + "_full_1.txt"
    path_2 = "../data/train_" + data + "_full_2.txt"
    path_3 = "../data/train_" + data + "_full_3.txt"
    path_4 = "../data/train_" + data + "_full_4.txt"
    path_5 = "../data/train_" + data + "_full_5.txt"
    path_6 = "../data/train_" + data + "_full_prep.txt"

    # set paths
    hs.set_paths(path_0, path_1)
    er.set_paths(path_1, path_2)
    ar.set_paths(path_2, path_3)
    sc.set_paths(path_3, path_4)
    sr.set_paths(path_4, path_5)
    wl.set_paths(path_5, path_6)

    # run
    hs.run()
    er.run()
    ar.run()
    sc.run()
    sr.run()
    wl.run()

    data = 'neg'
