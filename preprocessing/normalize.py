from preprocessing_interface import PreprocessingInterface
import enchant
from itertools import groupby

#normalizes each word in a tweet
# e.g  'llooooooovvvee' -> 'love'
# pip install --user pyenchant

class Normalize(PreprocessingInterface):

    def get_norm_string(self, substrings, ind):
        cur_string = ''.join(substrings)

        if self.d.check(cur_string):
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

    def test(self):
        self.d = enchant.Dict("en_US")
        #print(self.d.check('abcd'))
        words = ['lloooove','hello','aaabbcd']

        for word in words:
            # TODO do the split
            l = [''.join(g) for _, g in groupby(word)]
            string, is_word = self.get_norm_string(l,0)
            print(string)


    def run(self):
        super().run();

        self.d = enchant.Dict("en_US")

        # replace emoticons in input file
        output = open(self.output, 'w+')
        with open(self.input, mode='r') as input:
                for line in input:
                    for word in line.split():
                        print(word)
                        print(''.join(''.join(s)[:2] for _, s in itertools.groupby(word)))
                        if self.d.check(''.join(''.join(s)[:2] for _, s in itertools.groupby(word))):
                            output.write(''.join(''.join(s)[:2] for _, s in itertools.groupby(word)) + ' ')
                        else:
                            output.write(word + ' ')


                    output.write('\n')


n = Normalize()
n.set_paths('normtest.txt', 'normout.txt')
n.test()
