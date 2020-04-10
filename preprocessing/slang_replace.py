# Based on https://medium.com/@indreshbhattacharyya/remaking-of-shortened-sms-tweet-post-slangs-and-word-contraction-into-sentences-nlp-7bd1bbc6fcff
'''
Scrape slang words from https://www.noslang.com to handle words like 'omg',
'lol', 'dunno', etc.
Only replace words if they cannot be found in normal dict
'''

from preprocessing_interface import PreprocessingInterface
import enchant
from bs4 import BeautifulSoup
import urllib3
import os
import json

# TODO: loool, sooo, looove, hahaha, kiddin (etc.. 'in' instead of 'ing'), hasnt (etc no appostrophe), thankyou (spelled together)


class SlangReplace(PreprocessingInterface):

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


    def run(self):
        super().run();

        # init normal word checker
        eng_dict = enchant.Dict("en_US")

        # init slang dict
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_dict()

        with open(self.dict_path,'r', encoding='utf8') as file:
            self.slang_dict = json.loads(file.read())

        # replace slang words
        corr_word_rep = ['dunno']

        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
            for line in input:
                for word in line.split():
                    if word in corr_word_rep or (not eng_dict.check(word) and not word[0] == '#'):
                        if word in self.slang_dict:
                            output.write(self.slang_dict[word] + ' ')
                        else:
                            output.write(word + ' ')
