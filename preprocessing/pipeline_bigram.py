'''
Pipeline:
1. Format snap
2. Break up hash tags
3. Normalize
3. Remove appos
4. Remove stop words
5. Make bigram
'''

import os

from snap_format import SnapFormat
from hashtag_split_ww import HashtagSplitWW
from normalize import Normalize
from appos_remove import ApposRemove
from stopwords_remove import StopwordsRemove
from dict import Dict

class PipelineBigram:

    '''
    Give an array of input and output paths.
    The two array must have same length, every input file will be
    processed by pipeline and fully processed file will be written to
    output_path at same array index.
    '''
    def process(self, input_paths, output_paths):
        assert len(input_paths) == len(output_paths)

        # Init steps
        sf = SnapFormat(crop = 5000000)
        hs = HashtagSplitWW()
        no = Normalize()
        ar = ApposRemove()
        sr = StopwordsRemove()
        di = Dict(bigram_data=output)

        # execute pipeline
        for input_path, output_path in zip(input_paths, output_paths):

            # data paths
            path_tmp = os.path.dirname(input_path) + '/' + os.path.basename(input_path)[:-4] + '_tmp' + input_path[-4:]
            print(path_tmp)
            # set paths
            sf.set_paths(input_path, output_path)
            hs.set_paths(output_path, path_tmp)
            no.set_paths(path_tmp, output_path)
            ar.set_paths(output_path, path_tmp)
            sr.set_paths(path_tmp, output_path)

            # run
            print("starting")
            sf.run()
            print("format done")
            hs.run()
            print("hashtag split done")
            no.run()
            print("normalize done")
            ar.run()
            print("appo remove done")
            sr.run()
            print("stopword remove done")
            di.get_bigrams()
            print("bigram done")



# run it from here

file_path = os.path.dirname(os.path.abspath(__file__))
input = os.path.join(file_path, '../data/snap_1.txt')
output = os.path.join(file_path, '../data/snap_proc_out.txt')


p = PipelineBigram()
p.process([input], [output])
