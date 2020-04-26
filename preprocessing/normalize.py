'''
normalizes words
e.g  'llooooooovvvee' -> 'love'
'''

from preprocessing_interface import PreprocessingInterface
import enchant
from itertools import groupby
import os
import json

class Normalize(PreprocessingInterface):

    def __init__(self, dict_path='slang_dict.json'):
        self.slang_dict = {}
        file_path = os.path.dirname(__file__)
        self.dict_path = os.path.join(file_path, dict_path)


    def get_dict(self):
        # Generate links for all slang a-z
        linkDict=[]
        for one in range(97,123):
            linkDict.append(chr(one))

        # scrape sites
        http = urllib3.PoolManager()

        for alpha in linkDict:
            r = http.request('GET','https://www.noslang.com/dictionary/' + alpha)
            soup = BeautifulSoup(r.data,'html.parser')

            for i in soup.findAll('div',{'class':'dictionary-word'}):
                slang = i.find('abbr')['title']
                self.slang_dict[i.find('span').text[:-2]] = slang

        with open(self.dict_path, 'w') as file:
            json.dump(self.slang_dict, file)


    def is_word(self, string):
        return self.en_dict.check(string) or string in self.slang_dict


    def get_norm_string(self, substrings, ind):
        cur_string = ''.join(substrings)

        if self.is_word(cur_string):
            return cur_string, True

        elif ind == len(substrings):
            return cur_string, False

        elif len(substrings[ind]) > 1:
            # try replace with one letter
            substrings[ind] = substrings[ind][0]
            candidate, is_word = self.get_norm_string(substrings, ind+1)
            if is_word:
                return candidate, True

            # try replace with 2 letters
            substrings[ind] += substrings[ind][0]
            candidate, is_word = self.get_norm_string(substrings, ind+1)
            if is_word:
                return candidate, True

            # return unaltered string
            return cur_string, False

        else:
            return self.get_norm_string(substrings, ind+1)


    def run(self):
        super().run();

        # init english dict
        self.en_dict = enchant.Dict("en_US")

        # init slang dict
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_dict()

        with open(self.dict_path,'r', encoding='utf8') as file:
            self.slang_dict = json.loads(file.read())

        # normalize words
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                for word in line.split():
                    if not self.en_dict.check(word):
                        l = [''.join(g) for _, g in groupby(word)]
                        if len(l) <= 10:
                            word, _ = self.get_norm_string(l, 0)

                    output.write(word + ' ')

                output.write('\n')

        output.close()
