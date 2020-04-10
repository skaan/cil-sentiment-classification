from preprocessing_interface import PreprocessingInterface
import nltk
from nltk.corpus import stopwords


class StopwordsRemove(PreprocessingInterface):

    def run(self):
        super().run();

        # get stopwords
        try:
            nltk.data.find('stopwords')
        except LookupError:
            nltk.download('stopwords')

        stop_words = set(stopwords.words('english'))

        # remove
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        if not word in stop_words:
                            output.write(word + ' ')

                    output.write('\n')

        output.close()
