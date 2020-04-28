#inputfile = open('sample.txt')
#outputfile = open('sampleout.txt','w')
from preprocessing_interface import PreprocessingInterface
import re
import string

class SnapFormat(PreprocessingInterface):

    def __init__(self, crop=5000000):
        self.crop = crop*4

    def run(self):

        def isEnglish(string):
            english_check = re.compile(r'[a-z]')
            return len(english_check.findall(string)) == len(string)

        output = open(self.output, 'w+')
        with open(self.input, 'r', encoding='utf8') as inputfile:
            # ignore first 111 lines
            for i in range(111):
                next(inputfile)

            for i, line in enumerate(inputfile):
                line = line[2:].lower()
                if ((i%4 == 0) and (line != 'no post title\n')):
                    tmp = " ".join(filter(lambda x:x[0]!='@' and x[0:4]!='http' and isEnglish(x), line.split()))
                    tmp = tmp.translate(str.maketrans('','',string.punctuation))
                    tmp = ''.join([i for i in tmp if not i.isdigit()])
                    output.write(tmp + "\n")

                if i >= self.crop:
                    break

        output.close()
