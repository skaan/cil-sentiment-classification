from preprocessing_interface import PreprocessingInterface

import urllib.request
from bs4 import BeautifulSoup
import csv
import os


class EmoticonReplace(PreprocessingInterface):

    def __init__(self, dict_path='emoticon_dict.json'):
        file_path = os.path.dirname(__file__)
        self.dict_path =  os.path.join(file_path, dict_path)

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
        w = csv.writer(open(self.dict_path, 'w'))
        for i in range(len(emoticon)):
            w.writerow([emoticon[i], description[i]])


    def print_dict(self):
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_replace_words()
        else:
            with open(self.dict_path, mode='r') as f:
                reader = csv.reader(f)
                for rows in reader:
                    print(rows[0] + ", " + rows[1])

    def get_performance(self):
        print("Performance EmoticonReplace")

        # get dict size
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_replace_words()

        with open(self.dict_path, mode='r', encoding='utf8') as f:
            reader = csv.reader(f)
            for rows in reader:
                dict = {rows[0]:rows[1] for rows in reader}

        print("  Dict size: " + str(len(dict)))

        # process files
        import re
        from collections import defaultdict

        hit = defaultdict(int)
        missed = defaultdict(int)
        replaced = 0
        not_rec = 0

        regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        if word in dict:
                            hit[word] += 1
                            replaced += 1
                            output.write(dict[word] + ' ')
                        elif not regex.search(word) is None and len(word) > 1 and not word[0] == '#' and not (word[0] == '<' and word[-1] == '>'):
                            missed[word] += 1
                            not_rec += 1
                            output.write(word + ' ')
                        else:
                            output.write(word + ' ')

                    output.write('\n')

        output.close()

        print("  replaced: " + str(replaced))
        print("  not recognized: " + str(not_rec) + " (distinct: " + str(len(missed)) + ")")
        print()
        print("Emoticons not recognized:")
        print(missed)
        print()
        print("Emoticons replaced:")
        print(hit)


    def run(self):
        super().run();

        # scrape emoticon dict if not already done
        if not os.path.isfile(self.dict_path):
            print('scraping ...')
            self.get_replace_words()

        # get emoticon dict
        with open(self.dict_path, mode='r', encoding='utf8') as f:
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

        output.close()
