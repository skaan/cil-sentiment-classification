
# coding: utf-8

# Stemming and using the Porter stemmer
# =====================================
# 
# This exercise is about stemming.  In this exercise we do not ask you to
# implement a stemmer, but give you the opportunity to play with the Porter
# stemmer that is a part of the [NLTK python library](www.nltk.org).
# 
# Here is a quick example of how the Porter stemmer in NLTK can be used

# In[3]:


from nltk.stem.porter import *
stemmer = PorterStemmer()
words = [
    'caresses', 'flies', 'dies', 'mules', 'denied',
    'died', 'agreed', 'owned', 'humbled', 'sized',
    'meeting', 'stating', 'siezing', 'itemization',
    'sensational', 'traditional', 'reference', 'colonizer',
    'plotted'
]
for word in words:
    print(stemmer.stem(word))


# Feel free to play around with the Porter stemmer to get a feel for how it
# transforms words.
# 
# The required part for this exercise are the statements you find in moodle which touch on some issues that we face when we use a stemmer on the tokens of a document that we are indexing and searching for. Note that this question is exercise 2.1 from the book.
