
# coding: utf-8

# 
# Implementing a inverted index for n-words
# =========================================
# 
# In this week's exercise we will look at enhanced inverted indices.
# The first form we look at is the biword index (cf. section 2.4.1 of the book)
# and its generalized form the n-word index.
# 
# You can think of the biword index as an n-word index with `n = 2`.
# Similarly a triword index is identical to an n-word index with `n = 3`.
# Note that an n-word index with `n = 1` should be the same as a standard
# inverted index with respect to processing boolean queries.
# 
# In this notebook we provide you with a mostly complete reference
# implementation for the standard inverted index which you had to implement last
# week.
# 
# Your task is to modify `add_document()` to build an n-word index for a
# configurable n, and to modify `execute_query()` to be able to process phrase
# queries on the n-word index.
# A phrase query is a query such as "Romans countrymen" which should only return
# documents that contain this exact phrase.
# 
# We updated the provided parser to support phrases in arbitrary boolean
# queries.
# Any phrase needs to be enclosed in double quotes for the query parser to be
# able to detect it.
# The output format for a phrase is a list of the individual terms that make up
# the phrase, in order.
# Below we show a few example queries containing phrases

# In[ ]:


from queryparser import parse_query, process_ast
print(process_ast(parse_query('"Romans countrymen"')))
print(process_ast(parse_query('"Romans countrymen lovers"')))
print(process_ast(parse_query('"I think this"')))


# To make phrase queries more interesting we have modified the
# `tokenize_document()` function from last week to drop all the stop words
# listed in figure 2.5 in the book.
# If you're interested, you can acquire the list we use using the python snippet
# below.

# In[ ]:


from textutils import stop_words
print(stop_words)


# You can use the following function to remove stop words from your flattened queries:

# In[ ]:


def remove_stop_words(ast):
    print(ast)
    new_args = []
    if ast is not None:
        for a in ast.args:
            if isinstance(a, list):
                new_args.append([x for x in a if not x in stop_words])
            elif isinstance(a, Operation):
                new_args.append(remove_stop_words(a))
            elif a[0] == '-' and a[1:] in stop_words:
                pass
            elif a in stop_words:
                pass
            else:
                new_args.append(a)
        ast.args = new_args
    return ast


# Make sure that you implement a general n-word index with `n` as a global
# variable that you can use to change between different indices.
# 
# While we do not require you to be able to handle arbitrary boolean queries on
# your n-word index, think about how you would deal with queries that contain
# non-phrase operands in a more general purpose system.

# In[ ]:


# We keep the imports and global variables in a separate cell, so we can rerun
# the code cell without loosing the contents of the index.
from queryparser import parse_query, ParseException, process_ast, Operation
import glob
from textutils import tokenize_document
# global variables defining the index
documents = dict()
the_index = dict()
documentid_counter = 1
# the path to the corpus
corpuspath="../shared/corpus/*.txt"
# the n for n-word
n = 2


# Modify `add_document()` to build an n-word index for a configurable `n`. Use strings of `n` words, such as `"some words"`, as keys in the dictionary.

# In[ ]:


def add_document(path):
    '''
    Add a document to the inverted index. Return the document's document ID.
    Remember the mapping from document ID to document in the `documents`
    data structure.
    '''
    # make sure that we access the global variables we have defined
    global the_index, documents, documentid_counter, n
    # do not re-add the same document.
    if path in documents.values():
        # find and return document id for document which is already part of index.
        for docid, doc in documents.items():
            if doc == path:
                return docid
    docid = documentid_counter
    documents[docid] = path
    documentid_counter += 1
    print("Adding '%s' to index" % path)
    for word in tokenize_document(path):
        # TODO for assignment: change this inner loop to create a n-word index instead
        if word in the_index.keys() and the_index[word][-1] == docid:
            continue
        the_index.setdefault(word, []).append(docid)
    return docid


# Modify `execute_query()` to be able to process phrase queries on the n-word index. Note that all other methods below can remain untouched.
# As a first cut, don't worry about post-processing to weed out false-positives. You'll optionally be able to revisit your query processing to introduce post-processing later.

# In[ ]:


def negate(term):
    '''
    Negate postings list for `term`.  This is not feasible in a real-world
    system, but we utilize this for the fallback execution which is fairly
    naive.
    '''
    if term in the_index.keys():
        return sorted(set(documents.keys()) - set(the_index[term]))
    else:
        return list(documents.keys())

def execute_query_tree(flat):
    '''
    Fallback query execution for complex queries, we just recursively evaluate
    subtrees.
    '''
    result = set()
    if flat.op == 'AND':
        result = set(documents.keys())
    for arg in flat.args:
        # execute subtree etc
        if isinstance(arg, Operation):
            temp = execute_query_tree(arg)
        elif arg.startswith('-'):
            temp = negate(arg[1:])
        elif arg not in the_index.keys():
            print("NOTE: dropping term '%s' because no document contains it" % arg)
        else:
            temp = the_index[arg]

        if flat.op == 'OR':
            result = result | set(temp)
        elif flat.op == 'AND':
            result = result & set(temp)
        elif flat.op == 'NOT':
            assert(len(flat.args) == 1)
            result = set(documents.keys()) - set(temp)
    return sorted(result)

def intersect_two(p1, p2):
    '''
    Intersect two posting lists according to pseudo-code in Introduction
    to Information Retrieval, Figure 1.6
    '''
    answer = []
    while p1 != [] and p2 != []:
        if p1[0] == p2[0]:
            answer.append(p1[0])
            p1 = p1[1:]
            p2 = p2[1:]
        elif p1[0] < p2[0]:
            p1 = p1[1:]
        else:
            p2 = p2[1:]
    return answer

def intersect(terms):
    '''
    Intersect posting lists for a list of terms, according to pseudo-code
    in Introduction to Information Retrieval, Figure 1.7
    '''
    postings = [ the_index[t] if not t.startswith('-') else
                 negate(t[1:]) for t in terms ]
    # calculate word frequencies and sort term,freq pairs in ascending
    # order by frequency
    freqs = sorted([ (t, len(p)) for t,p in zip(terms, postings) ], key=lambda x: x[1] )
    terms, _ = map(list,zip(*freqs))
    if terms[0].startswith('-'):
        result = negate(terms[0][1:])
    else:
        result = the_index[terms[0]]
    terms = terms[1:]
    while terms != [] and result != []:
        if terms[0].startswith('-'):
            ps = negate(terms[0][1:])
        else:
            ps = the_index[terms[0]]
        result = intersect_two(result, ps)
        terms = terms[1:]
    return result

def execute_query(query):
    '''
    Execute a boolean query on the inverted index. We only support single
    operator queries ATM.  This method returns a list of document ids
    which satisfy the query in no particular order (i.e. the order in
    which the documents were added most likely :)).
    '''
    # We use a generated parser to transform the query from a string to an
    # AST.
    try:
        ast = parse_query(query)
    except ParseException as e:
        print("Failed to parse query '%s'\n" % query, e)
        return None

    # We preprocess the AST to flatten commutative operations, such as
    # sequences of ANDs. We also transform 'NOT <term>' arguments into
    # '-<term>' to allow smarter processing of AND NOT and OR NOT.
    flat = remove_stop_words(process_ast(ast))

    # Feel free to remove this print() if you don't find it helpful.
    print("Flat query repr:", flat)

    args = []
    # Perform pre-processing to expand n-word queries
    for i, arg in enumerate(flat.args):
        if isinstance(arg, list):
            # Assume it is a phrase
            for w in arg:
                assert(isinstance(w, str))

            # TODO: implement preprocessing for n-word queries
            # how exactly you do this depends on how you've constructed the
            # n-word index.
            print(\">>> Found phrase '%s' in query '%s', not yet implemented!\" % (' '.join(arg), query))
            assert(False)
            # hint:
            # if flat.op == 'AND':
            # ...
        else:
            args.append(arg)
            
    flat.args = args

    print("Flat query repr after pre-processing:", flat)
    
    for arg in flat.args:
        if isinstance(arg, Operation):
            # as soon as we find a argument to the top-level operation
            # which is not just a term, we fall back on the tree query
            # execution strategy.
            return execute_query_tree(flat)

    if flat.op == 'OR':
        # For demonstration purposes, utilize python's set() datatype 
        # to implement OR
        results = set()
        for arg in flat.args:
            if arg.startswith('-'):
                print("OR NOT not handled (query: '%s'" % query)
                return None
            else:
                results = results | set(the_index[arg])
        return sorted(results)

    elif flat.op == 'AND':
        return intersect(flat.args)

    elif flat.op == 'LOOKUP':
        assert(len(flat.args) == 1)
        if args[0] not in the_index.keys():
            # single term query for term not in vocabulary, return empty list
            # of document IDs
            return []
        else:
            # in this case the query was a single term
            return the_index[args[0]]
    else:
        print("Cannot handle query '%s', aborting..." % query)
        return None

def print_result(docs):
    '''
    Helper function to convert a list of document IDs back to file names
    '''
    if not docs:
        print("No documents found")
        print()
        return
    # If we got some results, print them
    for doc in docs:
        print('%d -> %s' % (doc, documents[doc]))
    print()


# The next cell allows us to build the index, and we run a test boolean query
# which the provided code should be able to answer satisfactorily.
# Because `add_document()` does not add documents that are already in the index,
# this cell can be run multiple times without adverse effects.

# In[ ]:


for file in glob.glob(corpuspath):
    add_document(file)


# Check the first 10 items in the index:

# In[ ]:


list(the_index.items())[:10]


# Below we provide a list of example phrase queries that your biword index should be able to handle.

# In[ ]:


# expected result: 
#   the_two_gentlemen_of_verona.txt
#   the_two_noble_kinsmen.txt
print_result(execute_query('"THE TWO"'))

# expected result: the_tragedy_of_julius_caesar.txt
print_result(execute_query('"Romans countrymen"')) 

# expected result: the_tragedy_of_julius_caesar.txt
print_result(execute_query('"Romans countrymen lovers"'))

# expected result:
#   king_henry_the_eighth.txt
#   the_tragedy_of_king_lear.txt
#   the_second_part_of_king_henry_the_sixth.txt
#   the_merchant_of_venice.txt
#   the_life_of_timon_of_athens.txt
#   the_tragedy_of_hamlet_prince_of_denmark.txt
#   alls_well_that_ends_well.txt
#   king_richard_the_third.txt
#   the_comedy_of_errors.txt
#   the_first_part_of_henry_the_sixth.txt
#   the_tragedy_of_antony_and_cleopatra.txt
#   pericles_prince_of_tyre.txt
#   the_first_part_of_king_henry_the_fourth.txt
#   the_winters_tale.txt
#   the_history_of_troilus_and_cressida.txt
#   as_you_like_it.txt
#   much_ado_about_nothing.txt
#   the_tragedy_of_othello_moor_of_venice.txt
print_result(execute_query('"I think this"'))


# If you just implement phrase queries on your n-word index as discussed in the book in section 2.4.1, you will notice that the query "I think this" returns a number of documents that do not actually contain the phrase "I think this" when we query a 2-word index.
# 
# A way to deal with this is to post-process the results of the query execution engine to drop documents which only contain partial phrases.
# 
# I decide to implement the postprocessing by calling out to the [`grep`
# tool](https://en.wikipedia.org/wiki/Grep) tool. What are potential problems with this postprocessing strategy?
# 
# Optional exercises
# ------------------
# 
# If you didn't feel challenged by this week's work, you can try your hand at
# doing (some of) the following implementation work:
# 
#  * implement a postprocessing step which actually correctly filters the
#    results
#  * Improve your query execution engine to handle phrases in conjunction with
#    boolean operators.
