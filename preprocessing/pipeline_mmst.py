'''
Pipeline:
1. Break up hash tags
2. Normalize
3. Contract
4. MMST spelling correction
'''

import os
import sys

from hashtag_split_ww import HashtagSplitWW
from appos_remove import ApposRemove
from normalize import Normalize
from wordnet_lemma import WordNetLemma
from remove_redundant import RemoveRedundant
from spelling_correction_mmst import SpellingCorrectionMMST
from emoticon_replace import EmoticonReplace
from slang_replace import SlangReplace
from spelling_correction_enchant import SpellingCorrectionEnchant
from remove_id import RemoveId

from dict import Dict


class PipelineMMST:

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


        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):

            # data paths
            path_0 = input_path
            path_1 = output_path[:-4] + '_1' + output_path[-4:]
            path_2 = output_path[:-4] + '_2' + output_path[-4:]
            path_3 = output_path


            # set paths
            hs.set_paths(path_0, path_1)
            nr.set_paths(path_1, path_2)
            ar.set_paths(path_2, path_3)

            # run
            print("starting with " + os.path.basename(input_path))
            hs.run()
            print(os.path.basename(input_path) + ": hashtag done.")
            nr.run()
            print(os.path.basename(input_path) + ": normalize done.")
            ar.run()
            print(os.path.basename(input_path) + ": appo done.")


# driver code
if __name__ == '__main__':
    pmmst = PipelineMMST()
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(file_path, '../data/mst_final2')

    inp_neg = os.path.join(data_folder, 'prep_neg.txt')
    inp_pos = os.path.join(data_folder, 'prep_pos.txt')
    inp_test = os.path.join(data_folder, 'prep_test.txt')

    out_neg = os.path.join(data_folder, 'train_neg.txt')
    out_pos = os.path.join(data_folder, 'train_pos.txt')
    out_test = os.path.join(data_folder, 'test.txt')

    pmmst.process([inp_neg, inp_pos, inp_test], [out_neg, out_pos, out_test])
