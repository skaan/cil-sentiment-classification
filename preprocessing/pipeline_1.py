'''
Pipeline:
1. Break up hash tags
2. Replace emoticons
3. Appos remove
 # 4. Remove stopwords
 # 5. Spelling correction
6. Wordnet lemmatize
'''

import os

from hashtag_split_ww import HashtagSplitWW
from emoticon_replace import EmoticonReplace
from appos_remove import ApposRemove
#from stopwords_remove import StopwordsRemove
#from spelling_correction import SpellingCorrection
from normalize import Normalize
from wordnet_lemma import WordNetLemma


class Pipeline1:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        # Init steps
        hs = HashtagSplitWW()
        er = EmoticonReplace()
        ar = ApposRemove()
        #sc = SpellingCorrection()
        nr = Normalize()
        #sr = StopwordsRemove()
        wl = WordNetLemma()

        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):

            # data paths
            path_0 = input_path
            path_1 = input_path[:-4] + '_p1_1' + input_path[-4:]
            path_2 = input_path[:-4] + '_p1_2' + input_path[-4:]
            path_3 = input_path[:-4] + '_p1_3' + input_path[-4:]
            path_4 = input_path[:-4] + '_p1_4' + input_path[-4:]
            path_5 = input_path[:-4] + '_p1_5' + input_path[-4:]
            path_6 = output_path


            # set paths
            hs.set_paths(path_0, path_1)
            er.set_paths(path_1, path_2)
            ar.set_paths(path_2, path_3)
            #sr.set_paths(path_3, path_4)
            #sc.set_paths(path_4, path_5)
            nr.set_paths(path_3, path_4)
            wl.set_paths(path_4, path_6)

            # run
            hs.run()
            er.run()
            ar.run()
            #sr.run()
            #sc.run()
            nr.run()
            wl.run()
