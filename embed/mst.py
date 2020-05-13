'''
Possible optimizations: Delaunay triangulation, C++.
'''

from collections import defaultdict
from math import sqrt
import enchant
from enchant.checker import SpellChecker
from nltk.corpus import stopwords
from embeddings import *
#Class to represent a graph
class MMST:

    def __init__(self, vertices):
        self.V = vertices
        self.adj_graph = [[] for i in range(vertices)]
        self.adj_mst = [[] for i in range(vertices)]
        self.del_cost = []


    ''' init graph'''
    def add_edge(self, u, v, w):
        self.adj_graph[u].append([v, w])
        self.adj_graph[v].append([u, w])

    def remove_node(self, node):
        for adj in self.adj_graph:
            for e in adj:
                if e[0] == node:
                    adj.remove(e)
                    #break
        self.adj_graph[node] = []

    # build graph from embedding vectors
    def build_graph_from_ebeddings(self, cand_vecs, corr_vecs):
        self.correct = len(corr_vecs)
        self.misspelled = len(cand_vecs)

        # first c node IDs for correct nodes
        # then d_i for candidates of i-th misspelled word
        self.surviving_candidates = []
        idx = len(corr_vecs)
        self.id_to_word = [idx]
        all_vecs = corr_vecs

        for cands in cand_vecs:
            self.surviving_candidates.append([*range(idx, idx+len(cands))])
            idx += len(cands)
            self.id_to_word.append(idx)
            all_vecs += cands

        print("id to word")
        print(self.id_to_word)

        # get all distances and insert edges
        last_word = 0
        next_word = self.id_to_word[0]
        word_idx = 0
        for i in range(len(all_vecs)):
            #for j in range(max(i+1, word_idx), len(all_vecs)):
            for j in range(i+1, last_word):
                dist = 0.0

                # get dist
                for k in range(len(all_vecs[0])):
                    dist += pow(all_vecs[i][k] - all_vecs[j][k], 2)

                dist = sqrt(dist)

                #insert edge
                self.add_edge(i, j, dist)

            for j in range(max(i+1, next_word),  len(all_vecs)):
                dist = 0.0

                # get dist
                for k in range(len(all_vecs[0])):
                    dist += pow(all_vecs[i][k] - all_vecs[j][k], 2)

                dist = sqrt(dist)

                #insert edge
                self.add_edge(i, j, dist)

            if i + 1 == next_word:
                last_word = next_word
                word_idx += 1

                if i+1 < len(all_vecs):
                    next_word = self.id_to_word[word_idx]


        print(self.adj_graph)


    # TODO: This is bad. just insert it into mst
    def get_cost(self, u, v):
        for e in self.adj_graph[u]:
            if e[0] == v:
                return e[1]
        return 0

    def is_mst_edge(self, u,v):
        for e in self.adj_mst[u]:
            if e == v:
                return True
        return False


    '''Build initial MST using Kruskal'''
    # find union parent
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
        edge_list = []
        for i, neighbors in enumerate(self.adj_graph):
            for j, w in neighbors:
                if i < j:
                    edge_list.append([i, j, w])

        edge_list = sorted(edge_list, key=lambda x: x[2])

        print('edge list: ' + str(len(edge_list)))

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
            u,v,w =  edge_list[curr_edge]
            curr_edge += 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # Take edge if that doesn't cause a cycle
            if x != y:
                edges_taken += 1
                self.adj_mst[u].append(v)
                self.adj_mst[v].append(u)
                self.union(parent, rank, x, y)



    ''' iteratively delete nodes from mst'''
    # reconstruct MST after deleting node v.
    # if change_gaph=true, MST will be altered and v will actually be deleted.
    # if change_graph=false, only the cost of that will be returned.
    def reconnect(self, node, change_graph=False):
        # if only one MST neighbour, just delete node and don't change anything.
        if len(self.adj_mst[node]) == 1:
            other = self.adj_mst[node][0]
            cost = -self.get_cost(node, other)

            if change_graph:
                self.adj_mst[node] = []
                self.adj_mst[other].remove(node)
                self.remove_node(node)

            return cost


        # ---- TODO: This is really badly done. look for conn. comp
        component = [-1] * self.V
        visited = [False] * self.V
        visited[node] = True
        queue = []

        comp = 0
        for start in range(self.V):

            if not visited[start]:
                queue.append(start)
                visited[start] = True

                while queue:
                    s = queue.pop(0)

                    for i in self.adj_mst[s]:
                        if not visited[i] and i != node:
                            component[i] = comp
                            queue.append(i)
                            visited[i] = True

                comp += 1
        ## ----


        # find cheapest outgoing edge for each disconn component
        cheapest_edges = []
        for other in self.adj_mst[node]:
            min_edge =[-1, -1, 10000]

            # do bfs on component to find cheapest outgoing edge
            visited = [False] * self.V
            queue = []

            queue.append(other)
            visited[other] = True

            while queue:
                s = queue.pop(0)

                for i, cost in self.adj_graph[s]:
                    if not visited[i] and i != node:
                        if self.is_mst_edge(s, i):
                            queue.append(i)
                            visited[i] = True
                        else:
                            if component[i] != component[s] and cost < min_edge[2]:
                                min_edge = [min(s,i), max(s,i), cost]

            cheapest_edges.append(min_edge)


        # delete node from MST
        del_cost = 0
        for other in self.adj_mst[node]:
            del_cost -= self.get_cost(node, other)
            if change_graph:
                self.adj_mst[other].remove(node)

        # delete node from graph
        if change_graph:
            self.remove_node(node)
            self.adj_mst[node] = []


        # delete duplicate edges
        cheapest_edges = sorted(cheapest_edges, key=lambda x: (x[0], x[1]))
        i = 1
        while i < len(cheapest_edges):
            if cheapest_edges[i][0] == cheapest_edges[i-1][0] and cheapest_edges[i][1] == cheapest_edges[i-1][1]:
                cheapest_edges.remove(cheapest_edges[i])
            else:
                i += 1


        # reconnect MST
        for e in cheapest_edges:
            del_cost += e[2]
            if change_graph:
                self.adj_mst[e[0]].append(e[1])
                self.adj_mst[e[1]].append(e[0])

        return del_cost


    # for each node, get cost of deleting
    def get_node_costs(self, nodes):
        self.del_cost = []
        for i in nodes:
            cost = self.reconnect(i, change_graph=False)
            self.del_cost.append([i, cost])

        self.del_cost = sorted(self.del_cost, key=lambda x: x[1])


    def get_word(self, node_id):
        for i, upper in enumerate(self.id_to_word):
            if upper > node_id:
                return i-1


    # TODO: Performance is terrible like this. Update del_cost when deleting
    def prune_mst(self):
        # always delete cheapest node that deletable.
        deletable = [*range(self.correct, self.V)]
        self.get_node_costs(deletable)

        print(self.adj_mst)
        print()
        print(deletable)

        #del_idx = 0
        while len(self.del_cost) > self.misspelled:
            del_node = self.del_cost[0][0]
            print('del node: ' + str(del_node))
            print('deletable: ', end='')
            print(deletable)
            deletable.remove(del_node)
            self.del_cost.remove(self.del_cost[0])
            word = self.get_word(del_node)
            print('surv cands: ', end='')
            print(self.surviving_candidates[word])
            if len(self.surviving_candidates[word]) > 1:
                print("delete")
                # delete
                self.reconnect(del_node, change_graph=True)
                self.get_node_costs(deletable)
                print('del cost: ', end='')
                print([d[0] for d in self.del_cost])
                self.surviving_candidates[word].remove(del_node)
                #del_idx = 0

            print('mst: ', end='')
            print(self.adj_mst)



# Driver code
stop_words = set(stopwords.words('english'))
stop_words.add('<user>')
stop_words.add('<url>')

d = enchant.Dict("en_US")

sentence = "the quck fx jumps over the fence"

stop_words = set(stopwords.words('english'))
stop_words.add('<user>')
stop_words.add('<url>')


# get non-stopwords and misspelled
print('Sentence:')
print(sentence)
print()
print('Candidates:')
correct = []
misspelled = []
for word in sentence.split():
    if d.check(word) and not word in stop_words:
        print('add to correct: ' + word)
        correct.append(word)
    elif not d.check(word):
        misspelled.append(d.suggest(word))

for sugs in misspelled:
    print(sugs)
print()


# visulazing
load = loader()
load.loadGloveModel('glove/glove.twitter.27B.25d.txt')

print("---------- Input:")

corr_in, words_t = load.get_emedding(correct)
print(words_t)

cand_in = []
for c in misspelled:
    vecs, words_t = load.get_emedding(c)
    print(words_t)
    cand_in.append(vecs)



print("--------------")
#print(corr_in)
#print(cand_in)

print("---------")

g = MMST(28)
#g = MMST(6)
g.build_graph_from_ebeddings(cand_in, corr_in)

g.build_mst()
g.prune_mst()
print(g.adj_mst)
