#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import re

import classifier_patterns
import numpy
import scipy
import sklearn
import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.utils
from scipy.spatial import distance
from sklearn.decomposition import LatentDirichletAllocation

import dataloader
import newsreader

pattern = re.compile(r"[^\x00-\x7F]+")
vocab_file, vocab_file_type = "vocab_pc_mac.pkl", "pickle"
embedding_file, embedding_dimensions, embedding_cache = "/Users/zxj/Downloads/glove.6B.50d.txt", 50, "embedding_mideast_guns.npz"

guns = "talk.politics.guns"
mideast = "talk.politics.mideast"
pc = "comp.sys.ibm.pc.hardware"
mac = "comp.sys.mac.hardware"
graphic = "comp.graphics"
auto = "rec.autos"

embedding_cache_pc_mac = "embedding_mac_ibm.npz"
embedding_cache_guns = "embedding_mideast_guns.npz"
embedding_cache_auto = "embedding_graphic_auto.npz"

option_list = [(guns, mideast, embedding_cache_guns),
                   (pc, mac, embedding_cache_pc_mac),
                   (graphic, auto, embedding_cache_auto)]


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def output_predict_measurement(y_development, prediction):
    print "accuracy_score", sklearn.metrics.accuracy_score(y_development, prediction)
    print "Precision:", sklearn.metrics.precision_score(y_development, prediction)
    print "Recall: ", sklearn.metrics.recall_score(y_development, prediction)
    print "F1:", sklearn.metrics.f1_score(y_development, prediction)


def calculate_svd_with_demension(demension):
    def calculate_svd(x_train):
        ur, sigmar, wrt = scipy.sparse.linalg.svds(x_train, k=demension)
        print ur.shape
        print sigmar.shape
        print wrt.shape
        print ur.dot(sigmar)
        t = x_train.dot(numpy.transpose(wrt))
        return t
    return calculate_svd


def pair_wise_similarity(expt):
    # this was too long to write out over and over
    # plus you want to be sure to have the same distance across the board
    # DRY!
    d = lambda *args: sklearn.metrics.pairwise.pairwise_distances(*args, metric='cosine')

    Xyes = expt.train_X[expt.train_y ==1, :]
    Xno = expt.train_X[expt.train_y != 1, :]
    one_minus = lambda a:  1-a
    vec_fuc = numpy.vectorize(one_minus)
    return vec_fuc(d(Xyes, Xno))


def predict_with_classifier_and_feature(train_info, development_info, classifier, vectorizer, reduction=None):
    X_train = vectorizer.fit_transform(train_info.data)
    y_train = train_info.target
    X_dev = vectorizer.transform(development_info.data)
    y_dev = development_info.target
    if reduction:
        X_train = reduction(X_train)
        X_dev = reduction(X_dev)

    _ = classifier.fit(X_train, y_train)
    pred = classifier.predict(X_dev)
    output_predict_measurement(y_dev, prediction=pred)


def remove_failed_documents(information):
    (new_data, removed_indices) = decode_list(information.data)
    information.target = numpy.delete(information.target, removed_indices)
    information.data = new_data


def decode_str(input):
    try:
        return str.decode(input)
    except ValueError as err:
        return ''


def decode_list(str_list):
    # type: (list) -> tuple(list, list)
    
    result_list = list()
    removed_indices = list()
    for index in xrange(len(str_list)):
        decoded_str = decode_str(str_list[index])
        if decoded_str:
            result_list.append(decoded_str)
        else:
            removed_indices.append(index)
    return result_list, removed_indices       


def remove_rebundatnt_elements(regex, input_list):
    for element in input_list:
        element = regex.sub("", element)
    return input_list


def centroid_distance(expt):
    if isinstance(expt.train_X, scipy.sparse.coo_matrix):
        expt.train_X = expt.train_X.tocsr()
    x_positive = expt.train_X[expt.train_y == 1, :]
    x_negative = expt.train_X[expt.train_y != 1, :]
    positive_mean = x_positive.mean(axis=0)
    negative_mean = x_negative.mean(axis=0)
    return 1 - scipy.spatial.distance.cosine(positive_mean, negative_mean)


def lda_similarity():
    news_train_info = sklearn.datasets.load_files(
        '/Users/zxj/PycharmProjects/nlp_project/20news-bydate-train')
    train_data = [decode_str(line) for line in news_train_info.data]
    train_data = [ele for ele in train_data if ele]
    tf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        stop_words='english', max_df=0.95, min_df=2, max_features=1000)
    x_train = tf_vectorizer.fit_transform(train_data)

    lda = LatentDirichletAllocation(n_topics=2, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    res = lda.fit_transform(x_train).transpose()
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, 20)
    distance = sklearn.metrics.pairwise.pairwise_distances(res, metric='cosine')
    print distance[0]


def reweight_features(expt) :
    Xyes = expt.train_X[expt.train_y ==1, :]
    Xno = expt.train_X[expt.train_y != 1, :]
    yesrates = numpy.log((Xyes.getnnz(axis=0) + 1.) / Xyes.shape[1])
    norates = numpy.log((Xno.getnnz(axis=0) + 1.) / Xno.shape[1])
    W = scipy.sparse.diags(yesrates - norates, 0)
    return lambda X, y: (X.dot(W), y)


def devide_array(x_array, y_array):
    (row, column) = x_array.shape
    first_category = list()
    second_category = list()
    for i in xrange(row):
        if y_array[i] == 1:
            first_category.append(x_array[i])
        else:
            second_category.append(x_array[i])
    return first_category, second_category


def read_train_data(folder, first, second):
    news_train_info = sklearn.datasets.load_files(folder, categories=[first, second])
    news_train_info.data = remove_rebundatnt_elements(pattern, news_train_info.data)
    return news_train_info


def stack_embeddings(embeddings):
    def operation(X, y):
        extra_features = X.shape[1] - embeddings.shape[0]
        if extra_features > 0 :
            Z = scipy.sparse.csr_matrix((extra_features, embeddings.shape[1]))
            W = scipy.sparse.vstack([embeddings, Z])
        else:
            W = embeddings
        return scipy.sparse.hstack([X, X.dot(W)]), y
    return operation


def document_embeddings(embeddings):
    def operation(X, y):
        extra_features = X.shape[1] - embeddings.shape[0]
        if extra_features > 0 :
            Z = scipy.sparse.csr_matrix((extra_features, embeddings.shape[1]))
            W = scipy.sparse.vstack([embeddings, Z])
        else:
            W = embeddings
        return X.dot(W), y
    return operation


def initial_experiment(data_loader, clssifier):
    expt1 = classifier_patterns.Experiment(data_loader, clssifier)
    expt1.initialize()
    return expt1


def add_embedding_feature(previous_expt, embeding, clssifier):
    expt_new = classifier_patterns.Experiment.transform(previous_expt,
                                                        stack_embeddings(embeding),
                                                        clssifier)
    expt_new.initialize()
    return expt_new


def fit_and_show_result(experiment):
    experiment.fit_and_validate()
    output_predict_measurement(experiment.dev_y, experiment.dev_predictions)
    print "centroid similarity  of group 1 is: {0}".format(centroid_distance(experiment))
    print "mean of pairwise similarity  of group 1 is: {0}".format(pair_wise_similarity(experiment).mean())
    print "max of pairwise similarity  of group 1 is: {0}".format(pair_wise_similarity(experiment).max())
    print "min of pairwise similarity  of group 1 is: {0}".format(pair_wise_similarity(experiment).min())
    print "----------------"

if __name__ == '__main__':
    tf_idf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
        stop_words='english', max_df=0.95, min_df=2)
    binary_vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        stop_words='english', max_df=0.95, min_df=2, binary=True, ngram_range=[1, 1])
    sgd = sklearn.linear_model.SGDClassifier(loss="log", penalty="elasticnet", n_iter=10)
    train_path = "/Users/zxj/Downloads/20news-bydate/20news-bydate-train/"
    test_path = "/Users/zxj/Downloads/20news-bydate/20news-bydate-test"

    (first_category, second_category, embedding_cache) = option_list[1]

    '''
    svd = TruncatedSVD(200)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    '''
    loader = dataloader.DataLoader(train_dir=train_path,
                                   test_dir=test_path, train_dir_pos=first_category,
                                   train_dir_neg=second_category,
                                   test_dir_pos=first_category, test_dir_neg=second_category,
                                   vectorizer=binary_vectorizer,
                                   )

    loader2 = dataloader.DataLoader(train_dir=train_path,
                                   test_dir=test_path, train_dir_pos=first_category,
                                   train_dir_neg=second_category,
                                   test_dir_pos=first_category, test_dir_neg=second_category,
                                   vectorizer=tf_idf_vectorizer,
                                   )
    loader.initialize()
    loader2.initialize()

    embedding_matrix = newsreader.load_sparse_csr(embedding_cache)

    first_experiemnt = initial_experiment(loader, sgd)
    expt1 = add_embedding_feature(first_experiemnt, embedding_matrix, sgd)
    second_experiment = classifier_patterns.Experiment.transform(first_experiemnt,
                                                                 reweight_features(first_experiemnt),
                                                                 sgd)
    print 'binary feature'
    fit_and_show_result(first_experiemnt)
    fit_and_show_result(expt1)

    print 'log count ratio '
    fit_and_show_result(second_experiment)

    expt2 = add_embedding_feature(second_experiment, embedding_matrix, sgd)
    fit_and_show_result(expt2)

    third_experiemnt = initial_experiment(loader2, sgd)
    print 'tf-idf'
    fit_and_show_result(third_experiemnt)

    expt4 = add_embedding_feature(third_experiemnt, embedding_matrix, sgd)
    fit_and_show_result(expt4)