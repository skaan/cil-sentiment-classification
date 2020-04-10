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
            for s in table.findAll('a'):
                s.replaceWithChildren()
            for s in table.findAll('sup', class_=True):
                s.extract()

            # add emoticons and description
            ignore = ['n/a', 'shocked', 'cup of tea']

            for row in table.findAll('tr'):
                cells = row.findAll('td')
                if len(cells) >= 3:
                    for i in range(len(cells)-2):
                        emoticon_string = cells[i].find(text=True).replace('\n', '').lower()
                        description_string = cells[-1].find(text=True).replace('\n', '').lower()

                        if description_string not in ignore:
                            single_emoticons = emoticon_string.split(' ')
                            for e in single_emoticons:
                                if len(e) != 0:
                                    emoticon.append(e)
                                    description.append(description_string)

        # clean
        for i in range(len(description)):
            # remove everything after ",", " or ", ". "
            description[i] = description[i].split(",", 1)[0]
            description[i] = description[i].split(" or ", 1)[0]
            description[i] = description[i].split(". ", 1)[0]

            # handle .. (some emoticons contain .. indicating symbol is repeated)
            if emoticon[i].endswith('..'):
                emoticon[i] = emoticon[i][:-2]
                emoticon.append(emoticon[i] + emoticon[i][-1]) # append with last sign once more
                description.append(description[i])
                emoticon.append(emoticon[-1] + emoticon[-1][-1]) # append with last sign twice more
                description.append(description[i])

            # add nose-less version of all emoticons: :-) -> :)
            no_nose = emoticon[i].split("‑")
            if len(no_nose) == 2:
                emoticon.append(no_nose[0] + no_nose[1])
                description.append(description[i])
            else:
                # Wikipedia used to different - for the noses
                no_nose = emoticon[i].split("-")
                if len(no_nose) == 2:
                    emoticon.append(no_nose[0] + no_nose[1])
                    description.append(description[i])

        # print
        for i in range(len(emoticon)):
            print(emoticon[i] + " " + description[i])

        # save
        w = csv.writer(open(dict_path, 'w'))
        for i in range(len(emoticon)):
            w.writerow([emoticon[i], description[i]])


    def print_dict(self):
        if not os.path.isfile(dict_path):
            print('scraping ...')
            self.get_replace_words()
        else:
            with open(dict_path, mode='r') as f:
                reader = csv.reader(f)
                for rows in reader:
                    print(rows[0] + ", " + rows[1])


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


er = EmoticonReplace()
er.print_dict()
