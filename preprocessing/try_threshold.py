from dict import Dict

d = Dict()
bigram = d.get_bigrams()

# just enter the word you want to try here
a = 'lady gaga'
b = 'iphone charger'
c = 'gasoline almonds'

print(bigram.get(a))
print(bigram.get(b))
print(bigram.get(c))
