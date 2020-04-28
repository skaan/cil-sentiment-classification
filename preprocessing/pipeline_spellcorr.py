'''
Pipeline:
1. Break up hash tags
2. Normalize
3. Tag remove
3. Spelling correction
'''

import os

from hashtag_split_ww import HashtagSplitWW
from normalize import Normalize
from tag_remove import TagRemove
from spelling_correction_enchant import SpellingCorrectionEnchant

class PipelineSc:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        assert len(input_paths) == len(output_paths)

        # Init steps
        hs = HashtagSplitWW()
        nr = Normalize()
        tr = TagRemove()
        sc = SpellingCorrectionEnchant()


        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):

            # data paths
            path_0 = input_path
            path_1 = os.path.dirname(input_path) + '/spell_corr/' + os.path.basename(input_path)[:-4] + '_1' + input_path[-4:]
            path_2 = os.path.dirname(input_path) + '/spell_corr/' + os.path.basename(input_path)[:-4] + '_2' + input_path[-4:]
            path_3 = os.path.dirname(input_path) + '/spell_corr/' + os.path.basename(input_path)[:-4] + '_3' + input_path[-4:]
            path_4 = output_path

            # set paths
            #hs.set_paths(path_0, path_1)
            #nr.set_paths(path_1, path_2)
            #tr.set_paths(path_2, path_3)
            sc.set_paths(path_3, path_4)

            # run
            #hs.run()
            #nr.run()
            #tr.run()
            sc.run_batch()




# run it from here

file_path = os.path.dirname(os.path.abspath(__file__))
data_neg = os.path.join(file_path, '../data/train_neg_full.txt')
data_pos = os.path.join(file_path, '../data/train_pos_full.txt')
test = os.path.join(file_path, '../data/test_data.txt')
data_neg_out = os.path.join(file_path, '../data/spell_corr/train_neg_full_sc.txt')
data_pos_out = os.path.join(file_path, '../data/spell_corr/train_pos_full_sc.txt')
test_out = os.path.join(file_path, '../data/spell_corr/test_data_sc.txt')


p = PipelineSc()
p.process([data_neg], [data_neg_out])
