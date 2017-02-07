
# coding: utf-8

# ## Short Illustration of wordnet programming in NLTK
#
# CS 533 NLP Spring 2017
#
# Matthew Stone
#
# The [NLTK documentation][1] gives a series of examples showing the information in [wordnet][2] and how to access it in python using NLTK.
#
# To complement this documentation, I found it useful to write some code that illustrates the wordnet functionality but also makes it convenient to explore wordnet yourself.
#
# [1]:http://www.nltk.org/howto/wordnet.html
# [2]:https://wordnet.princeton.edu/

# In[1]:

import nltk
from nltk.corpus import wordnet as wn


# Describe a word.  Print out all the wordnet senses associated with a word, along with their definition and an example sentence if there is one.
#
# Note that the `**kwargs` syntax enables you to pass optional arguments (like part of speech restrictions) into the wordnet synset lookup function.

# In[2]:

def describe(w, **kwargs) :
    for n in wn.synsets(w, **kwargs) :
        print n.name(), n.definition()
        if len(n.examples()) > 0 :
            print "     (", n.examples()[0], ")"


# List all the words associated with a wordnet synset.  Formally, the elements of wordnet synsets are [lemmas][1], so there's some work to be done to get comprehensible output.
#
# [1]:https://en.wikipedia.org/wiki/Lemma_(morphology)

# In[5]:

def synonyms(s, **kwargs) :
    ws = wn.synset(s, **kwargs)
    print "Synonyms in", ws.name()
    for l in ws.lemmas() :
        print "    ", str(l.name())


# Some examples

# In[6]:
if __name__ == '__main__':
    dog = wn.synset('dog.n.01')
    hyper = lambda s: s.hypernyms()
    hypo = lambda s: s.hyponyms()
    print dog.hyponyms()
    print list(dog.closure(hypo))
    handbag = wn.synset('bag.n.04')
