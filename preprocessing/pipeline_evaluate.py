from hashtag_split_ww import HashtagSplitWW
from appos_remove import ApposRemove
from spelling_correction import SpellingCorrection
from emoticon_replace import EmoticonReplace
from porter_stem import PorterStem

# data paths
path_0 = "../data/train_pos_full.txt"
path_1 = "../data/train_pos_full_er.txt"
path_2 = "../data/train_pos_full_er_ap.txt"

# init steps
er = EmoticonReplace()
er.set_paths(path_0, path_1)

ap = ApposRemove()
ap.set_paths(path_1, path_2)

# run pipeline
er.get_performance()
ap.get_performance()
