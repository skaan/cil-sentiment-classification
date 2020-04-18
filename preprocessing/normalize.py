from preprocessing_interface import PreprocessingInterface
import enchant
import itertools

#normalizes each word in a tweet 
# e.g  'llooooooovvvee' -> 'love'
# pip install --user pyenchant

class Normalize(PreprocessingInterface)
	def run(self):
		super().run();
		
		d = enchant.Dict("en_US")
		output = open(self.output, 'w+')
		with open(self.input, mode='r') as input:
			for tweet in input:
    		tweet=tweet.lower()
    		tweet=tweet.split()
    		for i in range(len(tweet)):
        	if d.check(''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i]))):
            tweet[i]=''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet[i]))
       	 else:
            tweet[i]=''.join(''.join(s)[:1] for _, s in itertools.groupby(tweet[i]))
    	tweet=' '.join(tweet)
