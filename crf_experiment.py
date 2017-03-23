# -*- coding: UTF-8 -*-

from xml.etree.ElementTree import parse
from xml.etree.ElementTree import Element
import re
import nltk
import sys
import crfutils
import argparse
import pycrfsuite
import tagtools
import sequence_tagging_patterns

parser = argparse.ArgumentParser(description='Process some integers.')
punct_pattern = re.compile(r'[\\.,:]+$')


def parse_dom(path, child_name):
    dom_tree = parse(path)
    root = dom_tree.getroot()
    '''type : Element'''
    result_list = list()
    for ele in root.iter(tag=child_name):
        ''':type : Element'''
        for index in xrange(len(ele)):
            attrib_map = {'tag': ele[index].tag, 'word': ele[index].text, 'F': []}
            result_list.append(map_to_list(attrib_map))
    return result_list


def contains_upper(token):
    b = False
    for c in token:
        b |= c.isupper()
    return b


def contains_lower(token):
    b = False
    for c in token:
        b |= c.islower()
    return b


def contains_alpha(token):
    b = False
    for c in token:
        b |= c.isalpha()
    return b


def contains_digit(token):
    b = False
    for c in token:
        b |= c.isdigit()
    return b


def get_shape(token):
    r = ''
    for c in token:
        if c.isupper():
            r += 'U'
        elif c.islower():
            r += 'L'
        elif c.isdigit():
            r += 'D'
        elif c in ('.', ','):
            r += '.'
        elif c in (';', ':', '?', '!'):
            r += ';'
        elif c in ('+', '-', '*', '/', '=', '|', '_'):
            r += '-'
        elif c in ('(', '{', '[', '<'):
            r += '('
        elif c in (')', '}', ']', '>'):
            r += ')'
        else:
            r += c
    return r


def map_to_list(input_map, tokenize=nltk.word_tokenize):
    word = input_map['word']
    word = punct_pattern.sub('', word.strip())
    words = tokenize(word)
    words_and_pos = nltk.pos_tag(words)
    return [{'word': ele, 'tag': input_map['tag'], 'F': [], 'pos': pos_info} for ele, pos_info in words_and_pos]


def binary(v):
    return 'yes' if v else 'no'


def feature_to_boolean(assert_function):
    def bool_string(word):
        return 'yes' if assert_function(word) else 'no'
    return bool_string

U = ['word', 'cu', 'cl', 'ca', 'cd', 'shape', 'pos', 'cy', 'booktitle', 'journal', 'page',
     'institution', 'tech']
B = ['word', 'shape', 'pos']
templates = []

uni_gram_map = {'cu': feature_to_boolean(contains_lower),
                'cl': feature_to_boolean(contains_upper),
                'ca': feature_to_boolean(contains_alpha),
                'cd': feature_to_boolean(contains_digit),
                'shape': get_shape,
                'cy': feature_to_boolean(sequence_tagging_patterns.contains_year),
                'booktitle': feature_to_boolean(sequence_tagging_patterns.probable_book_title),
                'journal': feature_to_boolean(sequence_tagging_patterns.probable_journal),
                'page': feature_to_boolean(sequence_tagging_patterns.probable_page),
                'institution': feature_to_boolean(sequence_tagging_patterns.probable_institution),
                'tech': feature_to_boolean(sequence_tagging_patterns.probable_tech)
                }


for name in U:
    templates += [((name, i),) for i in range(-1, 2)]
for name in B:
    templates += [((name, i), (name, i+1)) for i in range(-2, 1)]


def observation(v, defval=''):
    for key, uni_function in uni_gram_map.iteritems():
        v[key] = uni_function(v['word'])


def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
        # Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature


def generate_features(path, entry_name):
    entries = parse_dom(path, entry_name)
    for entry_list in entries:
        for entry in entry_list:
            observation(entry)
        feature_extractor(entry_list)
    return entries


def featute_list_to_map(feature_list):
    feature_map = {}
    for ele in feature_list:
        map_entry = ele.split('=')
        if map_entry and len(map_entry) == 2:
            feature_map[map_entry[0]] = map_entry[1]
    return feature_map


def train_model(train_path, output_path, entry_name):
    train_entries = generate_features(train_path, entry_name)
    x_train = [[ele['F'] for ele in train_entry] for train_entry in train_entries]
    y_train = [[ele['tag'] for ele in train_entry] for train_entry in train_entries]
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(x_train, y_train):
        trainer.append(xseq, yseq)
    
    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,  # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })
    trainer.train(output_path)


def predict(test_path, model_path, entry_name):
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    test_entries = generate_features(test_path, entry_name)
    x_test = [[ele['F'] for ele in train_entry] for train_entry in test_entries]
    correct_list = [ele['tag'] for train_entry in test_entries for ele in train_entry]
    predict_list = [tagger.tag(entry_list) for entry_list in x_test]
    predict_list = [item for sublist in predict_list for item in sublist]
    print tagtools.bieso_classification_report(correct_list, predict_list)


def chunking():
    if len(sys.argv) != 3:
        print 'number of arguments not sufficient'
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    entries = generate_features(input_path, 'entry')
    try:
        with open(output_path, 'w+') as output_file:
            crfutils.output_features(fo=output_file, X=entries, field='tag')
    except IOError as io_err:
        print 'Failed to open file {0}'.format(io_err.message)


def train_and_predict():
    train_path = '/Users/zxj/Downloads/data_set/reference_train.xml'
    model_path = './reference_all_features.model'
    test_path = '/Users/zxj/Downloads/data_set/reference_test.xml'
    train_model(train_path, model_path, 'entry')
    predict(test_path, model_path, 'entry')

if __name__ == '__main__':
    train_and_predict()