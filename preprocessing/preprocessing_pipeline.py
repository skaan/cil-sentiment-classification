from hashtag_split_ww import HashtagSplitWW
from spelling_correction import SpellingCorrection
from emoticon_replace import EmoticonReplace
from porter_stem import PorterStem

# data paths
path_0 = "../data/part_train_pos.txt"
path_1 = "path/after/hashtagsplit"
path_2 = "path/after/smileyreplace"
path_3 = "path/to/fully/preprocessed/data"


# init steps
hs = HashtagSplitWW()
hs.set_paths(path_0, path_1)

er = EmoticonReplace()
er.set_paths(path_1, path_2)

sc = SpellingCorrection()
sc.set_paths(path_2, path_3)


# run pipeline
hs.run()
er.run()
sc.run()
