from preprocessing_interface import PreprocessingInterface
from spellchecker import SpellChecker

class SpellingCorrection(PreprocessingInterface):

    def run(self):
        super().run();

        spell = SpellChecker()

        # correct
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():

                        if not (word[0] == '<' and word[-1] == '>'):
                            output.write(spell.correction(word) + ' ')
                        else:
                            output.write(word + ' ')

                    output.write('\n')

        output.close()

sc = SpellingCorrection()
sc.set_paths("../data/part_train_pos.txt", "corr.txt")
sc.run()
