'''
Pipeline:
1. remove duplicate
2. Space clean
'''

import os
import sys

from clean_spaces import CleanSpaces
from remove_redundant import RemoveRedundant
from remove_id import RemoveId
from tag_remove import TagRemove
from normalize import Normalize

from appos_remove import ApposRemove


from dict import Dict


class PipelineClean:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        # Init steps
        #rd = RemoveRedundant()
        cs = CleanSpaces()
        ap = ApposRemove()

        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):

            # data paths
            path_0 = input_path
            path_1 = output_path[:-4] + '_1' + output_path[-4:]
            path_2 = output_path[:-4] + '_2' + output_path[-4:]
            path_3 = output_path


            # set paths
            cs.set_paths(path_0, path_1)
            ap.set_paths(path_1, output_path)

            # run
            print("starting with " + os.path.basename(input_path))
            cs.run()
            print(os.path.basename(input_path) + ": tag remove done.")
            ap.run()
            print(os.path.basename(input_path) + ": red rem done.")
            #cs.run()
            #print(os.path.basename(input_path) + ": clean space done.")
            #tr.run()

# driver code
if __name__ == '__main__':
    pclean = PipelineClean()
    file_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(file_path, '../data/mst_first')

    inp_neg = os.path.join(data_folder, 'train_neg.txt')
    inp_pos = os.path.join(data_folder, 'train_pos.txt')
    inp_test = os.path.join(data_folder, 'test.txt')

    out_neg = os.path.join(data_folder, '../mst_first/ttrain_neg.txt')
    out_pos = os.path.join(data_folder, '../mst_first/ttrain_pos.txt')
    out_test = os.path.join(data_folder, '../mst_first/ttest.txt')

    pclean.process([inp_neg, inp_pos, inp_test], [out_neg, inp_pos, out_test])
