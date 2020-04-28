#inputfile = open('sample.txt')
#outputfile = open('sampleout.txt','w')
from preprocessing_interface import PreprocessingInterface
import re
import string

class parser(PreprocessingInterface):

    def run(self):
        output = open(self.output, 'a+')
        with open(self.input, 'r', encoding='utf8') as inputfile:
            next(inputfile)
            next(inputfile)

            i = 0
            for line in inputfile:
                if (i%4 == 0):
                    tmp = " ".join(filter(lambda x:x[0]!='@' and x[0:4]!='http', line.split()))
                    tmp = tmp.translate(str.maketrans('','',string.punctuation))
                    #space
                    #tmp = re.sub('([.,!?()])', r' \1 ', tmp)
                    #tmp = re.sub('\s{2,}', ' ', tmp)
                    output.write(tmp + "\n")
                i = i+1

parser = parser()
parser.set_paths('sample.txt','sampleout.txt')
parser.run()
