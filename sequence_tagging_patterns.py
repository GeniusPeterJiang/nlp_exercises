
# coding: utf-8

# ## Sequence Tagging Patterns
# Matthew Stone, CS 533, Spring 2017, to accompany second homework.
# 
# This notebook is designed to cover the basics and get you started with the second homework.  (There is a second version of this file for working with CRFsuite if you want to explore these problems further.)  You'll recognize a few of the [design patterns][1] here from last time, and maybe discover some new ones as well.
# 
# [1]:https://en.wikipedia.org/wiki/Software_design_pattern

# In[1]:

import nltk
import vocabulary
import itertools
import numpy as np
import scipy
import sklearn
import heapq
import tagtools
import re

# As before, we start with a setup that lets you customize your data resources.  Change what follows to show what data you want to use and where you've put it on your file system.

# In[2]:

reference_train_file = "/Users/zxj/Downloads/data_set/reference_train.xml"
reference_test_file = "/Users/zxj/Downloads/data_set/reference_test.xml"
reference_dev_file = "/Users/zxj/Downloads/data_set/reference_dev.xml"
reference_xml_item_keyword = "entry"

constrains_map = {'b': 'ie', 's': 'bs', 'i': 'ie', 'e': 'bs'}
month_set = {'january', 'february', 'march', 'april',
             'may', 'june', 'july', 'august', 'aug',
             'september', 'october', 'november', 'december'}
year_pattern = re.compile(r'^\(|19\d{2}|\)$')
page_pattern = re.compile(r'pages|pp|^\d{2,4}-\d{2,4}$')
volume_prefixes = {'Vol', 'No'}

conference_set = {'Conf', 'Conference', 'National', 'International', 'Symposium', 'Proc', 'Proceedings'}
journal_set = {'Transactions', 'Journal', 'Trans', 'Transaction'}
tech_set = {"Thesis", "Technical", "Tech", "RPT", "thesis"}
institution_pattern = re.compile(r'(Department|Dept|University|Laborator(y|ies)|Labs?)')

# ## Finding features in a sequence: Get a big window
# 
# Sequence tagging is a special problem because the correct interpretation of an individual token often depends on other tokens nearby.   So the key to learning powerful sequence taggers is to develop features that look beyond the target token to a meaningful window of tokens in the sequence.
# 
# The tagtools utilities include basic tokenizers that give back individual tokens, but here we compose those tokenizers with an aggregator operation `tagged_contexts` that gives access to the complete environment in which the target token appears.  
# 
# In particular, the output of `tagged_contexts` delivers each word along with the words that preceded it (in reverse order) and the words that follow it.  For example, to tag the word `'of'` in `['8', 'pounds', 'of', 'carrots']`, we get a preceding context of `['pounds', '8']` and a following context of `['carrots']`.  
# 
# You could adapt this kind of functionality to perform arbitrary preprocessing of the input stream.  For example, if you wanted the classifier to have access to part of speech tags for the input, you could run a part of speech tagger at this step.

# In[3]:

def tagged_contexts(seq) :
    '''take the tagged tokens in seq and create a new sequence 
    that yields the same tokens but presents them with the full
    context that precedes and follows them'''
    items = [x for x in seq]
    words = [w for (w,_) in items]
    for i, (w, t) in enumerate(items) :
        if i == 0:
            before = []
        else :
            before = words[i-1::-1]
        after = words[i+1:]
        yield (w, before, after), t


# ## Finding features in a sequence: Use open-ended patterns
# 
# The first part of this homework is to adapt the code to explore useful features for learning to assign tags to tokens.  It's convenient to separate the patterns that you write from the bookkeeping necessary to translate pattern occurrences into sparse vectors on particular tokens.  This function `make_cxt_feature_processor` does the trick.
# 
# ### How the code works
# 
# It takes two arguments: `word_tests`, a list of word-level patterns, and `list_tests`, a list of list_level patterns.  Each element of each list is a function:
# 
#     f: subitem, indicator -> feature_name option
#     
# That is, each function takes a `subitem` to process (either a word or a list from the context of a target token) and an `indicator` string capturing the environment in which `subitem` occurs relative to the target token, and returns a string `feature_name` recording the way `subitem` matches the pattern encoded by `f` in this context if it does, or `None` otherwise.  Examples of such functions are below. 
# 
# The `make_cxt_feature_processor` function then returns a `feature_processor` function that can be used by the data manager to construct the sparse feature vector for a particular target `item`.  This feature processor constructs a series of `subitem` targets based on the `item` context (right now it uses the target itself, and the preceding and following token if any, along with the full list of preceding and following tokens).  It then applies the `word_tests` to the word subitems, and the `list_tests` to the list subitems, and encodes the matching features as integers using the passed `features` vocabulary.
# 
# ### What you might want to change
# 
# This abstract method is flexible enough to handle any patterns that you want to apply to the subitems it finds.  However, you may want to change this function to identify additional subitems for feature processing.  For example, you might want to scan the contexts for significant nearby tokens in a different way (looking at two nearby words, or skipping punctuation).  In addition, you will need to extend the interface if you want to look at bigram features or other features that look at multiple subitems simultaneously; these kinds of patterns need to be specified with a different functional form.

# In[4]:

def make_cxt_feature_processor(word_tests, list_tests) :
    def feature_processor(features, item):
        this, before, after = item
        result = []
        
        def addf(name) :
            if name :
                r = features.add(name)
                if r :
                    result.append(r)

        def add_word_features(w, code) :
            for ff in word_tests :
                addf(ff(w, code))

        def add_list_features(l, code) :
            for ff in list_tests :
                addf(ff(l, code))

        first = lambda l: l[0] if l else None
        
        add_word_features(this, u"w")
        add_list_features(before, u"-")
        add_list_features(after, u"+")
        for wx, cx in [(first(before), u"-w"),
                       (first(after), u"+w")] :
            if wx:
                add_word_features(wx, cx)
    
        return np.array(result)
    return feature_processor


# ## Writing specific feature detectors
# 
# ### Model Features
# 
# Here are some model features to give a sense of what's involved in writing the pattern matching routines for a feature processor.
# 
# * The `identity_feature` constructs a feature for the item itself.  It shows how you can incorporate material from the item into the feature definition you create on a match.  Token identity is always a good default for memorizing arbitrary associations in a classification problem.
# 
# * The `all_digits` feature tests whether a word item consists entirely of numerical characters, e.g., matches `[0-9]*`  In the bibliography context, this might be a good starting cue to recognize a date or for the volume number of a journal.  In the recipe domain this might be a good cue for recognizing the quantity of an ingredient.
# 
# * The `lonely_initial` feature tests whether something looks like an abbreviated first name, which of course is useful for identifying author and editor fields in bibliographies: a two character token consisting of an upper case letter and a period (e.g,. `A.`)
# 
# * The `is_empty` feature tests whether a list context contains no tokens.  When applied to the context features for a target token, this feature fires in one way when the target token is the first token in a text sequence and in another way when the target token is the final token in a text sequence.  This makes it generally useful for identifying material that's consistently placed at the beginning or end of descriptions (e.g., author vs date in bibliography entries, or quantity vs comment in recipe elements).
# 
# ### Suggestions: directions and methodology
# 
# You'll want to expand this section by adding lots of other features.  Some ideas, in no particular order:
# 
# - Features based on the general shape of a token (capitalization, acronyms) or specific patterns (e.g., email or web address).
# 
# - Features that look at the length and constituency of the target token.  For example, number strings are likely to play a different role in a bibliography if they are a single digit (volume), three digits (pages) or four digits (year).
# 
# - Features that look for character classes mixed with distinctive punctuation tokens, suggesting for example a page range (bibliography) or a fraction (recipes).
# 
# - Features that access wordnet or other external resources to get a hint about the semantic category of a word (trying to get a handle on institutions, locations, dates, quantities, comments, journal names, etc).
# 
# - Features that exploit additional preprocessing that you've done of the target string (for example, part of speech tags).  Note that you'll have to do a bunch of refactoring if you make general changes to the way tokens are represented.
# 
# - Features that look for particular keywords at rule-governed places in the nearby context.  For example, in a bibliography, you get cues on the location of publication information from the location of the keyword `In`.
# 
# Play a bit.  Your experience crafting features will have a big role in allowing you to appreciate the opportunities and challenges for improving machine learning algorithms as well as the direction current research in the field is taking.  You'd be surprised how many interesting lists you can roll into your algorithm with a quick google search or by scraping content from wikipedia.  On the other hand, you can also be surprised at how difficult it can be to get a machine learning algorithm to make the right decision on a particular token even if it seems like it ought to have plenty of information in the available features for the answer to be obvious.
# 
# A good strategy is to build your pipeline and inspect items in the development set.  You can use the `tagutils` `DataManager` method `test_features_dev_item` to inspect the way that the classifier is representing a particular text from the dev set in terms of the features you've provided.  (This can let you see whether some aspect of the text that's important for the classification decision is not captured in the features so far.  It can also show you whether you have a bug in the feature detection code, which of course is a common failure mode for machine learning methods.)  You can compare this with the predictions that you make using the `tagutils` function `visualize`, or once you create a `TaggingExperiment` object (defined below), using the `visualize_classifier` or `visualize_decoder` methods, to get some insights into what tokens your model is treating incorrectly and thus where you are not representing the tokens and their context sufficiently richly or precisely for the learning algorithm to understand them.

# In[5]:

identity_feature = lambda item, code: "{}: {}".format(code, item)


def all_digits(item, code):
    if item.isdigit():
        return u"{}: is all digits".format(code)


def single_digit(input_str):
    return input_str.isdigit() and len(input_str) == 1


def double_digit(input_str):
    return input_str.isdigit() and len(input_str) == 2


def multidigit(input_str):
    return input_str.isdigit() and len(input_str) >= 3



def lonely_initial(item, code) :
    if len(item) == 2 and item[0].isupper and item[1] == '.' :
        return u"{}: lonely initial".format(code)


def is_empty(l, code):
    if not l:
        return u"{}: empty".format(code)


def feature_formatter(feature_assert, input_template):
    def formatted_word_feature(item, code):
        if feature_assert(item):
            return input_template.format(code)
    return formatted_word_feature


def is_month(input_str):
    # type: (str) -> bool
    return input_str.lower() in month_set


def contains_year(input_str):
    if year_pattern.search(input_str):
        return True
    else:
        return False


def probable_page(input_str):
    if page_pattern.search(input_str):
        return True
    else:
        return False


def probable_book_title(input_str):
    return input_str in conference_set


def probable_tech(input_str):
    return input_str in tech_set


def probable_journal(input_str):
    return input_str in journal_set


def contains_volume_prefixes(input_str):
    return input_str in volume_prefixes


def probable_institution(input_str):
    if institution_pattern.search(input_str):
        return True
    else:
        return False

# ## Putting your data pipeline together
# 
# The data manager here is slightly more complicated than what we used for text classification.  We need:
# * XML file names for train, test and dev sets
# * the keyword to access individual items in the XML file
# * a tokenizer, mapping each item DOM object to a sequence of `(item, tag)` tuples.  Remember, each `item` shoud be represented with all the structure necessary to identify relevant features.  This sample tokenizer returns the target word together with its context, as described above.
# * a token view, which controls how the token items will be visualized for interaction.  The basic view here just returns the target word, stripped of the contextual information we are maintaining
# * a feature processor, which converts an item into a sparse feature representation
# * a constructor that builds a vocabulary for the feature processor
# 
# The most important thing that you will likely wind up changing here is the `feature_processor` code.  You'll see the two lists of features being used (the word features and the list features).  As you add more features, you will need to add those features to these arguments so they actually get checked against tokens in the input.

# In[6]:

default_tokenizer = lambda i: tagged_contexts(tagtools.bies_tagged_tokens(i))

default_token_view = lambda i : i[0]

default_feature_processor = make_cxt_feature_processor(
    [all_digits, lonely_initial,
     identity_feature, feature_formatter(is_month, u'{0}: is month'),
     feature_formatter(contains_year, u'{0}: contains year'),
     feature_formatter(contains_volume_prefixes, u'{0}: contains volume prefixes'),
     feature_formatter(probable_page, u'{0}: is probable a page'),
     feature_formatter(probable_book_title, u'{0}: is a book title'),
     feature_formatter(probable_journal, u'{0}: is a journal'),
     feature_formatter(probable_tech, u'{0}: is a tech'),
     feature_formatter(probable_institution, u'{0}: is an institution'),
     feature_formatter(single_digit, u'{0}: is a  digit with length 1'),
     feature_formatter(double_digit, u'{0}: is a digit with length 2'),
     feature_formatter(multidigit, u'{0}: is a digit with length larger than 3')
     ], [is_empty])


def default_features(vocab) :
    return lambda data: vocab


def create_bib_data():

    bib_features = vocabulary.Vocabulary()

    bib_data = tagtools.DataManager(reference_train_file,
                                    reference_test_file,
                                    reference_dev_file,
                                    reference_xml_item_keyword,
                                    default_tokenizer,
                                    default_token_view,
                                    default_features(bib_features),
                                    default_feature_processor)
    bib_data.initialize()
    return bib_data, bib_features


def is_start_or_single(input_str):
    if not input_str:
        return False
    if input_str[0] == 's' or input_str[0] == 'b':
        return True
    return False


def is_intermediate_or_end(input_str):
    if not input_str:
        return False
    if input_str[0] == 'i' or input_str[0] == 'e':
        return True
    return False


class TaggingExperiment(object) :
    '''Organize the process of getting data, building a classifier,
    and exploring new representations'''
    
    def __init__(self, data, features, classifier, decoder) :
        'set up the problem of learning a classifier from a data manager'
        self.data = data
        self.classifier = classifier
        self.features = features
        self.decoder = decoder
        self.initialized = False
        self.trained = False
        self.decoded = False
        
    def initialize(self) :
        'materialize the training data, dev data and test data as matrices'
        if not self.initialized :
            self.train_X, self.train_y, self.train_d = self.data.training_data()
            self.features.stop_growth()
            self.dev_X, self.dev_y, self.dev_d = self.data.dev_data()
            self.test_X, self.test_y, self.test_d = self.data.test_data()
            self.initialized = True
        
    def fit_and_validate(self) :
        'train the classifier and assess predictions on dev data'
        if not self.initialized :
            self.initialize()
        self.classifier.fit(self.train_X, self.train_y)
        self.tagset = self.classifier.classes_
        self.trained = True
        self.dev_predictions = self.classifier.predict(self.dev_X)
        self.accuracy = sklearn.metrics.accuracy_score(self.dev_y, self.dev_predictions)
    
    def visualize_classifier(self, item_number) :
        'show the results of running the classifier on text number item_number'
        if not self.trained :
            self.fit_and_validate()
        w = self.data.dev_item_token_views(item_number)
        s = self.dev_d[item_number]
        e = self.dev_d[item_number+1]
        tagtools.visualize(w, {'actual': self.dev_y[s:e], 
                               'predicted': self.dev_predictions[s:e]})

    def decode_and_validate(self) :
        '''use the trained classifier and beam search to find the consistent
        analyses of all the items in the dev data'''
        if not self.trained :
            self.fit_and_validate()
        self.dev_log_probs = self.classifier.predict_log_proba(self.dev_X)
        results = []
        self.dev_partials = []
        self.dev_exacts = []
        for i in range(len(self.dev_d)-1) :
            s = self.dev_d[i]
            e = self.dev_d[i+1]
            tags, score = self.decoder.search(self.tagset, self.dev_log_probs[s:e])
            p_t = tags[1:-1]

            results.append(p_t)

            self.dev_partials.append(tagtools.agrees(p_t, iter(self.dev_y[s:e]), partial=True))
            self.dev_exacts.append(tagtools.agrees(p_t, iter(self.dev_y[s:e]), partial=False))
        self.dev_decoded = np.concatenate(results)

    def decode_and_validate2(self, data_x, data_d, data_y):
        '''use the trained classifier and beam search to find the consistent
        analyses of all the items in the dev data'''
        if not self.trained :
            self.fit_and_validate()
        log_probs = self.classifier.predict_log_proba(data_x)
        results = []
        self.dev_partials = []
        self.dev_exacts = []
        for i in range(len(data_d)-1) :
            s = data_d[i]
            e = data_d[i+1]
            tags, score = self.decoder.search(self.tagset, log_probs[s:e])
            p_t = tags[1:-1]

            results.append(p_t)

            self.dev_partials.append(tagtools.agrees(p_t, iter(data_y[s:e]), partial=True))
            self.dev_exacts.append(tagtools.agrees(p_t, iter(data_y[s:e]), partial=False))
        return np.concatenate(results)


    def visualize_decoder(self, item, index_list, actual, predictions) :
        'show the results of running the classifier and decoder on text number item_number'
        if not self.decoded :
            self.decode_and_validate()
        w = self.data.dev_item_token_views(item)
        s = index_list[item]
        e = index_list[item+1]
        tagtools.visualize(w, {'actual': actual[s:e],
                               'predicted': predictions[s:e]})

    @classmethod
    def transform(cls, expt, operation, classifier) :
        'use operation to transform the data from expt and set up new classifier'
        if not expt.initialized :
            expt.initialize()
        result = cls(expt.data, classifier)
        result.train_X, result.train_y, result.train_d = operation(expt.train_X, expt.train_y, expt.train_d)
        result.dev_X, result.dev_y, result.dev_d = operation(expt.dev_X, expt.dev_y, expt.dev_d)
        result.test_X, result.test_y, result.test_d = operation(expt.test_X, expt.test_y, expt.test_d)
        result.initialized = True
        return result


# ## Searching for Consistent Taggings
# 
# The beam search decoder uses two constructs to model sequence structure.  * First, there are `states`, which correspond to tags.  In the case of `BIES` tags, these tags actually incorporate a lot of information about what has come before and what will come later, which needs to be taken into account in decoding.
# * Second, there is the `status`, which summarizes the analysis that has been developed so far in a meaningful way.  The status lets you make sure that each new thing that you recognize represents new information (if appropriate).
# 
# The beam search decoder uses a set of abstractions to monitor the process of integrating the tags assigned to individual tokens into a consistent and meaningful overall analysis.  These functions manipulate the `state` and `status` data structures that the decoder uses.
# 
# * `initial_status()`: Function that returns the `status` corresponding to an empty text.  You shouldn't make this empty; the beam search decoder uses empty status as a signal of failure.  Our default alternative, which you probably won't have to change, just indicates that we're at the beginning of the text.
# 
# * `is_consistent(t1, t2)`: Indicates whether tag `t2` can sensibly follow `t1`.  This is designed to encode the constraints of the `BIES` tag regime.  For example, if you have a tag `i: author` it means you're in the middle of an author specification; the only thing that can meaningfully come after that is another `i: author` tag, continuing the author definition, or an `e: author` tag indicating the end of the author segment.  You can't have a `b: ...` or `s: ...` tag or anything other than an `... author` tag.  **It is part of the homework for you to fill in the details here yourself in a sensible way.**
# 
# * `next_status(status, t1, t2)`: Creates the new status that you get when you start from `status` in state `t1` and update to `t2`.  In general, your status will indicate the information that you've found and the information you still need to analyze the text in a consistent way.  So when you move from `t1` to `t2` you may have to check whether `t2` represents the beginning of new information that makes sense in context or not (if not, probably you should return `None`).  And it should update the `status` to reflect everything you've now seen.  **Again the default implementation is inadequate, and you need to fill this in for yourself.  You need to think about the structure of recipes and bibliography items to do this correctly.  Can something have multiple comments in a recipe?  Definitely.  Can it have multiple quantities?  Probably not.  Can something have two authors?  Probably not.  In fact, something probably can't have both a journal and a booktitle field: where did this article actually appear?**
# 
# * `do_special(status, t1, j, node, heuristics)`.  You may want to deal with model failure.  For example, if you look at the dev set in the bibliography data set, you will be able to find a couple of items that actually present duplicate publication information (for example, an article that appeared first in a conference, then appeared in one or two later edited collections).  This is a qualitatively different kind of bibliography entry.  You want to have constraints describing normal bibliography entries, but you want to just bail on those constraints when you get an exceptional entry, and recognize as much as you can.  That's what the `do_special` method handles.  The method is called to consider the possibility of model failure given status `status` and state `t1` at position `j`.  If you think there's model failure, you need to construct an analysis node convering the remaining tokens in the stream (`None` is a reasonable label for the remaining tokens; there are `len(heuristics)-2` tokens in total, and you're about to handle token `j`), and you need to assign a score to it (which should penalize the score derivable from `heuristics` according to some fixed penalty).  Actually, you may be able to construct this analysis in a few different ways, so this method returns a list of `(node, score)` pairs.  **You can get a fair amount better precision by using this method, but obviously it requires a pretty solid understanding of the role this method plays in decoding.**
# 
# Your job is to flesh this out in a sensible way.  The definitions below simply enable the decoder to return exactly the same analysis as the independent classifier.  But if you update these definitions to do something sensible, you can get a meaningful improvement in performance.

# In[12]:

# a good default
def initial_status():
    return frozenset(['START'])


# terrible.  work out the right answer for yourself
def everything_is_consistent(t1, t2) :
    return True


def modified_consistent(t1, t2):
    if t1 == 'START' or t2 == 'END':
        return True
    first = t1[0]
    second = constrains_map[first]
    if t2[0] not in second:
        return False

    if t1[0] in 'bi' and t1[3:] != t2[3:]:
        return False

    return True


# doesn't enforce unique specification.  sometimes you should.
def status_whats_seen(status, t1, t2):
    field = t2[3:]
    return status.union([field])


def next_status_ref(status, t1, t2):
    unique_set = {'title', 'date', 'author'}
    exclusive_set = {'booktitle', 'journal', 'tech'}
    if 'note' not in status:
        unique_set = unique_set.union(exclusive_set)

    field = t2[3:]

    if is_start_or_single(t2) and field in status:
        if field in unique_set:
            return None
        if field in exclusive_set:
            exclusive_set.discard(field)
            if not exclusive_set.isdisjoint(status):
                return None
            
    return status.union([field])


# enforce model constraints all the time.  you can do better.
def dont_do_special(status, t1, j, node, heuristics, punishment):
    return []


def do_special_ref(status, t1, j, node, heuristics, punishment):
    end_set = {'author', 'title'}
    publication_set = {'booktitle', 'journal', 'tech'}

    if publication_set.isdisjoint(status) or not end_set.issubset(status):
        return []

    if not end_set.issubset(status):
        return []

    length_limit = len(heuristics) - 2

    if j > length_limit:
        return []

    new_node = node
    for index in xrange(j, len(heuristics) - 2):
        new_node = (None, new_node)

    score = heuristics[j + 1] + punishment
    return [(new_node, score)]


def compare_predict_and_actual(predict, actual):
    compare = zip(predict, actual)
    for first, second in compare:
        print first + "\t" + second
    print '-----------------'


# ## Setting up experiments
# 
# The definitions below put everything together.  We create a classifier to learn correlations between features and tags, and crate a decoder to put the learned decisions together into an analyses of complete texts.  Then we set up the infrastructure to explore the results systematically.

# In[13]:
def create_bib(data, features, next_status, do_special):
    bib_classifier = sklearn.linear_model.SGDClassifier(loss="log",
                                           penalty="elasticnet",
                                           n_iter=5)

    bib_decoder = tagtools.BeamDecoder(initial_status,
                                       modified_consistent,
                                       next_status,
                                       do_special)

    bib = TaggingExperiment(data,
                            features,
                            bib_classifier,
                            bib_decoder)

    bib.initialize()
    return bib

if __name__ == '__main__':
    bib_data, bib_features = create_bib_data()
    bib = create_bib(bib_data, bib_features, next_status_ref, do_special_ref)
    #bib.decode_and_validate()

    decoded_test = bib.decode_and_validate2(bib.test_X, bib.test_d, bib.test_y)
    print tagtools.bieso_classification_report(bib.test_y, decoded_test)
