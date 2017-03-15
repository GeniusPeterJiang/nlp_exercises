# -*- coding: UTF-8 -*-

from xml.etree.ElementTree import parse
from xml.etree.ElementTree import Element
import tagtools
import re
import nltk
import sys
import crfutils
import argparse
import pycrfsuite

parser = argparse.ArgumentParser(description='Process some integers.')


def parse_dom(path, child_name):
    dom_tree = parse(path)
    root = dom_tree.getroot()
    '''type : Element'''
    entry_list = list()

    for ele in root.iter(tag=child_name):
        ''':type : Element'''
        for index in xrange(len(ele)):
            attrib_map = {'tag': ele[index].tag, 'word': ele[index].text, 'F': []}
            entry_list.append(attrib_map)

    return entry_list


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


def binary(v):
    return 'yes' if v else 'no'

U = ['word', 'cu', 'cl', 'ca', 'cd']
B = ['word']
templates = []

for name in U:
    templates += [((name, i),) for i in range(-1, 2)]
for name in B:
    templates += [((name, i), (name, i+1)) for i in range(-2, 2)]


def observation(v, defval=''):
    # Contains a uppercase letter.
    v['cu'] = binary(contains_upper(v['word']))
    # Contains a lowercase letter.
    v['cl'] = binary(contains_lower(v['word']))
    # Contains a alphabet letter.
    v['ca'] = binary(contains_alpha(v['word']))
    # Contains a digit.
    v['cd'] = binary(contains_digit(v['word']))
    # Contains a symbol.


def feature_extractor(X):
    # Apply attribute templates to obtain features (in fact, attributes)
    crfutils.apply_templates(X, templates)
    if X:
        # Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature


def generate_features(path, entry_name):
    entries = parse_dom(path, entry_name)
    for ele in entries:
        observation(ele)
    feature_extractor(entries)
    return entries


def featute_list_to_map(feature_list):
    feature_map = {}
    for ele in feature_list:
        map_entry = ele.split('=')
        if map_entry and len(map_entry) == 2:
            feature_map[map_entry[0]] = map_entry[1]
    return feature_map


def train_and_test():
    path = '/Users/zxj/Downloads/data_set/reference_train.xml'
    test_path = '/Users/zxj/Downloads/data_set/reference_test.xml'
    train_entries = generate_features(path, 'entry')
    test_entries = generate_features(test_path, 'entry')
    x_train = [ele['F'] for ele in train_entries]
    y_train = [ele['tag'] for ele in train_entries]
    x_test = [ele['F'] for ele in test_entries]


def train_model(train_path, output_path, entry_name):
    train_entries = generate_features(train_path, entry_name)
    x_train = [ele['F'] for ele in train_entries]
    y_train = [ele[entry_name] for ele in train_entries]
    trainer = pycrfsuite.Trainer(verbose=False)
    trainer.append(x_train, y_train)
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
    x_test = [ele['F'] for ele in test_entries]
    predict_list = [tagger.tag(ele) for ele in x_test]
    correct_list = [ele[entry_name] for ele in test_entries]
    print("Predicted:", ' '.join(predict_list))
    print("Correct:  ", ' '.join(correct_list))


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

if __name__ == '__main__':
    chunking()