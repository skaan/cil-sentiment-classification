import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class loader():

	def loadGloveModel(self,File):
		print("loading glove model")
		f = open(File,'r')
		gloveModel = {}
		for line in f:
			splitLines = line.split()
			word = splitLines[0]
			wordEmbedding = np.array([float(value) for value in splitLines[1:]])
			gloveModel[word] = wordEmbedding
		print(len(gloveModel)," words loaded!")
		self.model = gloveModel
		
	def find_closest_embeddings(self, vector): # doesnt work for now
		if self.model is None:
			raise Exception("load vectors first")
		#print(spatial.distance.euclidean(self.model["hi"], vector))
		return sorted(self.model.keys(), key = lambda word: spatial.distance.euclidean(self.model[word], vector))
		
	def plotSNE(self, words):
		tsne = TSNE(n_components=2, random_state=0)
		vectors = [self.model[word] for word in words]
		Y = tsne.fit_transform(vectors)
		plt.scatter(Y[:,0],Y[:,1])
		for label, x, y in zip(words, Y[:,0], Y[:,1]):
			plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords="offset points")
		plt.show()
		
		
load = loader()
load.loadGloveModel('glove/glove.twitter.27B.25d.txt')
#res = load.find_closest_embeddings(load.model["king"])[1:6]  
#print(res)
words = "sister brother man woman uncle aunt"
words = words.split()
load.plotSNE(words)
