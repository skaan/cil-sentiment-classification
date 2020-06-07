'''
Pipeline:
1. Break up hash tags
2. Normalize
3. Appos remove
4. MST spelling correction
'''

import os
import sys

from hashtag_split_ww import HashtagSplitWW
from appos_remove import ApposRemove
from normalize import Normalize
from wordnet_lemma import WordNetLemma
from spelling_correction_mst import SpellingCorrectionMST

from dict import Dict

class PipelineMST:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        # Init steps
        hs = HashtagSplitWW()
        nr = Normalize()
        ar = ApposRemove()
        sc = SpellingCorrectionMST()

        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):

            # data paths
            path_0 = input_path
            path_1 = output_path[:-4] + '_mst_1' + output_path[-4:]
            path_2 = output_path[:-4] + '_mst_2' + output_path[-4:]
            path_3 = output_path[:-4] + '_mst_3' + output_path[-4:]
            path_4 = output_path


            # set paths
            hs.set_paths(path_0, path_1)
            nr.set_paths(path_1, path_2)
            ar.set_paths(path_2, path_3)
            sc.set_paths(path_3, path_4)


            # run
            print("starting with " + os.path.basename(input_path))
            hs.run()
            print(os.path.basename(input_path) + ": hashtag split done.")
            nr.run()
            print(os.path.basename(input_path) + ": normalize done.")
            ar.run()
            print(os.path.basename(input_path) + ": appo remove done.")
            sc.run()
            print("done with " + os.path.basename(input_path))


# driver code
if __name__ == '__main__':
    pmst = PipelineMST()
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(file_path, '../data/raw')
    inp_neg = os.path.join(data_folder, 'train_neg_full.txt')
    inp_pos = os.path.join(data_folder, 'train_pos_full.txt')
    out_neg = os.path.join(data_folder, '../mst_first/train_neg.txt')
    out_pos = os.path.join(data_folder, '../mst_first/train_pos.txt')
    pmst.process([inp_neg, inp_pos], [out_neg, out_pos])
