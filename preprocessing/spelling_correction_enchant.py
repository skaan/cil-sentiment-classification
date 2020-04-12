from preprocessing_interface import PreprocessingInterface

import enchant
from enchant.checker import SpellChecker

class SpellingCorrectionEnchant(PreprocessingInterface):

    def run(self):
        super().run();

        spell = SpellChecker("en_UK","en_US")

        # correct
        output = open(self.output, 'w+')
        with open(self.input, mode='r', encoding='utf8') as input:
            for line in input:
                spell.set_text(line)
                for err in spell:
                    if len(err.suggest()) > 0:
                        sug = err.suggest()[0]
                        err.replace(sug)

                line = spell.get_text()
                output.write(line)

        output.close()
