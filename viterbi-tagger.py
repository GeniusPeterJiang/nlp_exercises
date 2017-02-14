import nltk
import math
from nltk.corpus import brown
from nltk.collocations import *


class Experiment(object):
    pass


def brown_sentence_items() :
    for sent in brown.tagged_sents(tagset='universal'):
        yield ('START', 'START')
        for (word, tag) in sent :
            yield (tag, word)
        yield ('END', 'END')


def calculate_taged_sequence_probability(expt, ts):
    result = expt.cpd_tagwords[ts[0][0]].prob(ts[0][1])
    for index in xrange(len(ts) - 1):
        result *= expt.cpd_tags[ts[index][0]].prob(ts[index + 1][0]) * \
                  expt.cpd_tagwords[ts[index + 1][0]].prob(ts[index + 1][1])
    return result


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def pmi_estimation_hmm(expt, bigram):
    (t1, w1), (t2, w2) = bigram
    pmi = expt.cpd_tags[t1].pro(t2) * expt.tag_fd.N() / expt.tag_fd[t2]
    return math.log(pmi, 2)


def main():
    expt1 = Experiment()
    expt1.cfd_tagwords = nltk.ConditionalFreqDist(brown_sentence_items())
    expt1.cpd_tagwords = nltk.ConditionalProbDist(expt1.cfd_tagwords, nltk.MLEProbDist)

    expt1.cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams((tag for (tag, word) in brown_sentence_items())))
    expt1.cpd_tags = nltk.ConditionalProbDist(expt1.cfd_tags, nltk.MLEProbDist)
    expt1.tagset = set((tag for (tag, word) in brown_sentence_items()))

    ts1 = [("PRON", "I"), ("VERB", "want"), ("PRT", "to"), ("VERB", "race")]
    ts2 = [("PRON", "I")]
    prob_tagsequence = expt1.cpd_tagwords["PRON"].prob("I") * \
                       expt1.cpd_tags["PRON"].prob("VERB") * expt1.cpd_tagwords["VERB"].prob(
        "want") * \
                       expt1.cpd_tags["VERB"].prob("PRT") * expt1.cpd_tagwords["PRT"].prob("to") * \
                       expt1.cpd_tags["PRT"].prob("VERB") * expt1.cpd_tagwords["VERB"].prob("race")
    print "The probability of the tag sequence ' PRON VERB PRT VERB ' for 'I want to race' is:", prob_tagsequence
    prob_tagsequence_2 = calculate_taged_sequence_probability(expt1, ts1)
    print "The probability of the tag sequence ' PRON VERB PRT VERB ' for 'I want to race' is:", prob_tagsequence_2

def collocation():
    expt1 = Experiment()
    expt1.tag_fd = nltk.FreqDist(tag for (tag, word) in brown_sentence_items())
    expt1.wt_fd = nltk.FreqDist(i for i in brown_sentence_items())
    expt1.bigram_fd = nltk.FreqDist(nltk.bigrams(i for i in brown_sentence_items()))
    expt1.finder = BigramCollocationFinder(expt1.wt_fd, expt1.bigram_fd)
    expt1.finder.apply_freq_filter(4)
    expt1.bigram_measures = nltk.collocations.BigramAssocMeasures()
    collocs = expt1.finder.score_ngrams(expt1.bigram_measures.pmi)
    for ele in collocs[:5]:
        print ele

if __name__ == '__main__':
    collocation()