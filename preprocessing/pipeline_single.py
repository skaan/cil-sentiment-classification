'''
Pipeline:
1. remove duplicate
2. Space clean
'''

import os
import sys

from hashtag_split_ww import HashtagSplitWW
from normalize import Normalize
from slang_replace import SlangReplace
from appos_remove import ApposRemove
from emoticon_replace import EmoticonReplace
from spelling_correction_mst import SpellingCorrectionMST

from mst_pre import MSTPre

from dict import Dict


class PipelineSingle:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        # Init steps
        step = MSTPre()
        #step = SpellingCorrectionMST()

        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):

            # set paths
            step.set_paths(input_path, output_path)
            #step.set_paths(output_path+'pre', output_path)

            # run
            print("starting with " + os.path.basename(input_path))
            step.run()
            #step.run()
            print("done")


# driver code
if __name__ == '__main__':
    pmst = PipelineSingle()
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(file_path, '../data/mst_final2')

    inp_neg = os.path.join(data_folder, 'prep_neg.txt')
    inp_pos = os.path.join(data_folder, 'prep_pos.txt')
    inp_test = os.path.join(data_folder, 'prep.txt')

    out_neg = os.path.join(data_folder, '../mst_final2/train_neg.txt')
    out_pos = os.path.join(data_folder, '../mst_final2/train_pos.txt')
    out_test = os.path.join(data_folder, '../mst_final2/test.txt')

    pmst.process([inp_neg, inp_pos, inp_test], [out_neg, out_pos, out_test])
