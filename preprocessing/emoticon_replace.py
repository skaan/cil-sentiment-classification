from preprocessing_interface import PreprocessingInterface

import urllib.request
from bs4 import BeautifulSoup
import csv
import os

dict_path = 'emoticon_dict.csv'

class EmoticonReplace(PreprocessingInterface):

    def get_replace_words(self):
        # get wikipedia tables
        url = "https://en.wikipedia.org/wiki/List_of_emoticons"
        page = urllib.request.urlopen(url)

        soup = BeautifulSoup(page, "lxml")
        tables = soup.findAll('table', class_='wikitable')[:-3]
        tables.remove(tables[3])

        # extract emoticon and description
        emoticon = []
        description = []

        for table in tables:
            # remove all links
            for ab in table.findAll('a'):
                ab.replaceWithChildren()
            for s in table.findAll('sup', class_=True):
                s.extract()

            # add emoticons and description
            for row in table.findAll('tr'):
                cells = row.findAll('td')
                if len(cells) >= 3:
                    for i in range(len(cells)-2):
                        # Innerhalb einer zelle noch mit span
                        # Innerhalb von span mit abstand
                        emoticon_string = cells[i].find(text=True).replace('\n', '')
                        if len(emoticon_string) != 0:
                            emoticon.append(emoticon_string.lower())
                            description.append(cells[-1].find(text=True).lower())

        # TODO clean
        emoticon.append(':d')
        description.append('smile')

        # print
        for i in range(len(emoticon)):
            print(emoticon[i] + " " + description[i])

        # save
        w = csv.writer(open(dict_path, 'w'))
        for i in range(len(emoticon)):
            w.writerow([emoticon[i], description[i]])



    def run(self):
        super().run();

        # scrape emoticon dict if not already done
        if not os.path.isfile(dict_path):
            print('scraping ...')
            self.get_replace_words()

        # get emoticon dict
        with open(dict_path, mode='r') as f:
            reader = csv.reader(f)
            dict = {rows[0]:rows[1] for rows in reader}

        # replace emoticons in input file
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        if word in dict:
                            output.write(dict[word] + ' ')
                        else:
                            output.write(word + ' ')

                    output.write('\n')
