'''
Possible optimizations: Delaunay triangulation, C++.
'''

# params TODO: Print words, ret embedding, take embedder and words in build from emb
#  algo params: Max dist, popularity to dist
#
#


from collections import defaultdict
from math import sqrt
import enchant
from enchant.checker import SpellChecker
from nltk.corpus import stopwords
from embeddings import *
#Class to represent a graph
class MMST:

    def __init__(self, vertices=0, popularity_fac=0, max_dist_fac=-1):
        self.V = vertices
        self.adj_graph = [[] for i in range(vertices)]
        self.adj_mst = [[] for i in range(vertices)]

        self.popularity_fac = popularity_fac
        self.max_dist_fac = max_dist_fac

        self.del_cost = []

        self.sorted_edges = []

        self.node_to_word = {}


    '''graph functions'''
    def add_edge(self, u, v, w):
        self.adj_graph[u].append([v, w])
        self.adj_graph[v].append([u, w])


    def remove_node(self, node):
        # remove from graph
        for adj in self.adj_graph[node]:
            self.adj_graph[adj[0]] = [e for e in self.adj_graph[adj[0]] if e[0] != node]
        self.adj_graph[node] = []

        # remove from mst
        for adj in self.adj_mst[node]:
            self.adj_mst[adj[0]] = [e for e in self.adj_mst[adj[0]] if e[0] != node]
        self.adj_mst[node] = []

        # remove from sorted edges
        self.sorted_edges = [e for e in self.sorted_edges if e[0] != node and e[1] != node]


    def distance_sqr(self, a, b):
        d = 0.0
        for i in range(len(a)):
            d += pow(a[i] - b[i], 2)
        return d


    def is_mst_edge(self, u,v):
        for e in self.adj_mst[u]:
            if e == v:
                return True
        return False


    # embed words and build graph
    def build_graph_from_words(self, embedder, correct, candidates, verbose=True):
        # convert words to embeddings
        # Keep track of: Which node elem which canset, node -> word
        # Init: Survivors of a candset
        node_count = 0
        self.surviving_candidates = []

        embs, words = embedder.get_emedding(correct)
        self.correct = len(words)
        self.candset_borders = [len(words)]
        for c in candidates:
            embs_c, words_c = load.get_emedding(c)
            embs += embs_c
            words += words_c
            self.candset_borders.append(len(words))
            self.surviving_candidates.append([*range(self.candset_borders[-2], self.candset_borders[-1])])
        self.candsets = len(self.candset_borders) - 1

        for i, word in enumerate(words):
            self.node_to_word[i] = word

        if verbose:
            print('built graph with the following words:')
            print(words, end='\n\n')


        # init graph
        self.V = self.candset_borders[-1]
        self.adj_graph = [[] for i in range(self.V)]
        self.adj_mst = [[] for i in range(self.V)]
        self.del_cost = []


        # add edges from correct nodes to all nodes
        for i in range(self.correct):
            for j in range(i+1, self.V):
                dist_sqr = self.distance_sqr(embs[i], embs[j])
                self.add_edge(i, j, dist_sqr)


        # add edges between candsets
        for i in range(len(self.candset_borders)-1):
            for j in range(self.candset_borders[i], self.candset_borders[i+1]):
                for k in range(self.candset_borders[i+1], self.V):
                    dist_sqr = self.distance_sqr(embs[j], embs[k])
                    self.add_edge(j, k, dist_sqr)



    '''pretty prints'''
    def pprint_adjecency(self, graph, weights=False):
        for i, l in enumerate(graph):
            print(i, end=':    ')
            if weights:
                print(l)
            else:
                print([n[0] for n in l])


    def print_mst_words(self):
        for i, adj in enumerate(self.adj_mst):
            if len(adj) > 0:
                print(self.node_to_word.get(i), end=', ')
        print()



    '''Build initial MST using Kruskal'''
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])


    # join 2 unions
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller tree to bigger tree
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1


    # Construct MST
    def build_mst(self):
        # get list of edges and sort according to weight
        self.sorted_edges = []
        for i, neighbors in enumerate(self.adj_graph):
            for j, w in neighbors:
                if i < j:
                    self.sorted_edges.append([i, j, w])

        self.sorted_edges = sorted(self.sorted_edges, key=lambda x: x[2])

        # Create union for each node
        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Join unions
        curr_edge = 0
        edges_taken = 0
        while edges_taken < self.V -1 :

            # Look at smallest edge still available
            u,v,w =  self.sorted_edges[curr_edge]
            curr_edge += 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # Take edge if that doesn't cause a cycle
            if x != y:
                edges_taken += 1
                self.adj_mst[u].append([v, w])
                self.adj_mst[v].append([u, w])
                self.union(parent, rank, x, y)



    ''' iteratively delete nodes from mst'''
    # reconstruct MST after deleting node v.
    # if change_gaph=true, MST will be altered and v will actually be deleted.
    # if change_graph=false, only the cost of that will be returned.
    def reconnect(self, node, change_graph=False):
        # if only one MST neighbour, just delete node and don't change anything.
        if len(self.adj_mst[node]) == 1:
            cost = -self.adj_mst[node][0][1]

            if change_graph:
                self.remove_node(node)

            return cost

        cost = 0.0

        # get connected components (= build unions for kruskal)
        parent = [-1] * self.V
        rank = [0] * self.V
        visited = [False] * self.V
        queue = []

        for start, w in self.adj_mst[node]:
            cost -= w

            queue.append(start)
            visited[start] = True
            parent[start] = start

            while queue:
                s = queue.pop(0)

                for i, _ in self.adj_mst[s]:
                    if not visited[i] and i != node:
                        parent[i] = start
                        rank[start] += 1
                        queue.append(i)
                        visited[i] = True

        # remove node
        edges_missing = len(self.adj_mst[node]) - 1
        if change_graph:
            self.remove_node(node)


        # run last steps of kruskal to rejoin connected components
        curr_edge = 0
        edges_taken = 0

        while edges_taken < edges_missing:
            # Look at smallest edge still available
            u,v,w =  self.sorted_edges[curr_edge]
            curr_edge += 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # Take edge if that doesn't cause a cycle
            if x != y:
                edges_taken += 1
                cost += w
                if change_graph:
                    self.adj_mst[u].append([v, w])
                    self.adj_mst[v].append([u, w])
                self.union(parent, rank, x, y)

        return cost


    # for each node, get cost of deleting
    def get_node_costs(self, nodes):
        self.del_cost = []
        for i in nodes:
            cost = self.reconnect(i, change_graph=False)
            self.del_cost.append([i, cost])

        self.del_cost = sorted(self.del_cost, key=lambda x: x[1])


    def get_surviving_candidates(self, node_id):
        for i, upper in enumerate(self.candset_borders):
            if upper > node_id:
                return self.surviving_candidates[i-1]


    def build_mmst(self):
        self.build_mst()

        deletable = [*range(self.correct, self.V)]
        self.get_node_costs(deletable)

        print(deletable)

        # always delete cheapest node that deletable.
        cand_selected = 0
        while cand_selected < self.candsets:
            del_node, _ = self.del_cost.pop()
            #print('del node: {}'.format(del_node))
            deletable.remove(del_node)
            #print(deletable)

            surv_cands = self.get_surviving_candidates(del_node)
            if len(surv_cands) > 1:
                # delete
                self.reconnect(del_node, change_graph=True)
                self.get_node_costs(deletable)
                surv_cands.remove(del_node)
            else:
                cand_selected += 1

# Driver code
# set dicts
stop_words = set(stopwords.words('english'))
stop_words.add('<user>')
stop_words.add('<url>')

d = enchant.Dict("en_US")

# input sentences
sentences = ["the quck fox jumps over the new fnce"]


# init embedder
load = loader()
load.loadGloveModel('glove/glove.twitter.27B.25d.txt')

# feed sentences
for sentence in sentences:
    print("--------------------------------")
    print('Sentence:')
    print(sentence)

    # remove stopwords, split into correct and misspelled
    correct = []
    misspelled = []
    for word in sentence.split():
        if d.check(word) and not word in stop_words:
            correct.append(word)
        elif not d.check(word):
            misspelled.append([w.lower() for w in d.suggest(word)])

    print('\nCandidates:')
    for sugs in misspelled:
        print(sugs)
    print()

    # init graph
    g = MMST()
    g.build_graph_from_words(load, correct, misspelled)
    g.build_mmst()
    g.print_mst_words()
