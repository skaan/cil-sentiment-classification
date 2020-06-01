from preprocessing_interface import PreprocessingInterface
import enchant
from dict import Dict
import multiprocessing as mp
import threading
from math import floor

class Manager(PreprocessingInterface):

    def __init__(self):
        self.sentences = []
        self.corrected = [""]
        self.nb = 0
        self.cores = mp.cpu_count()
        self.load = loader()
        self.load.loadGloveModel('../embed/glove/glove.twitter.27B.25d.txt')

    def writeout(self):
        output = open(self.output, 'w+')
        for line in self.corrected:
            output.write(line + '\n')

    def readin(self):
        with open(self.input, mode='r') as input:
            for line in input:
                self.sentences += [line]
                self.nb += 1
        self.corrected *= nb

    def checker(self, pre,post, id):
        g = MMST()
        for x in range(pre,post):
            tmp = g.input_sentences(sentences[x], self.load, verbose=False)
            corrected[x] = tmp

    def run(self):
        super().run()
        
        self.readin()
        
        # dictionnary defined in MMST __init___
        share = floor(self.nb / self.cores)
        
        ts = [threading.Thread(target=self.checker, args=(i*share, min((i+1)*share,self.nb), i,)) for i in range(self.cores)]

        for t in ts:
            t.start()
        for t in ts:
            t.join()

        self.writeout()
