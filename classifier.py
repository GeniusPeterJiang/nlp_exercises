import sklearn
import numpy
import scipy

import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.metrics
import sklearn.utils
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def output_predict_measurement(y_development, prediction):
    print "accuracy_score", sklearn.metrics.accuracy_score(y_development, prediction)
    print "Precision:", sklearn.metrics.precision_score(y_development, prediction)
    print "Recall: ", sklearn.metrics.recall_score(y_development, prediction)
    print "F1:", sklearn.metrics.f1_score(y_development, prediction)


def vectorize(train_info, vectorizer):
    x_train = vectorizer.fit_transform(train_info.data)
    y_train = train_info.target
    return x_train, y_train


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


def predict_with_classifier_and_feature(train_info, development_info, classifier, vectorizer, reduction=None):
    X_train, y_train = vectorize(train_info, vectorizer)
    X_dev = vectorizer.transform(development_info.data)
    y_dev = development_info.target
    if reduction:
        X_train = reduction(X_train)
        X_dev = reduction(X_dev)

    _ = classifier.fit(X_train, y_train)
    pred = classifier.predict(X_dev)
    output_predict_measurement(y_dev, prediction=pred)


def predict_with_classifier_and_feature2(train_info, development_info, classifier, vectorizer):
    X_train, y_train = vectorize(train_info, vectorizer)
    X_dev = vectorizer.transform(development_info.data)
    y_dev = development_info.target
    svd1 = calculate_svd_with_demension(20)
    X_train = svd1(X_train)
    X_dev = svd1(X_dev)
    _ = classifier.fit(X_train, y_train)
    pred = classifier.predict(X_dev)
    output_predict_measurement(y_dev, prediction=pred)


def compare_feature_extraction(classifier):
    reviews_train_info = sklearn.datasets.load_files('reviews/train')
    reviews_dev_info = sklearn.datasets.load_files('reviews/dev')
    tf_idf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')
    svd = TruncatedSVD(100)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    svd1 = calculate_svd_with_demension(20)

    print
    predict_with_classifier_and_feature2(train_info=reviews_train_info,
                             development_info=reviews_dev_info,
                             classifier=classifier,
                             vectorizer=tf_idf_vectorizer)


def get_feature_set(train_info):
    tf_idf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')
    X_train, y_train = vectorize(train_info, tf_idf_vectorizer)
    return set(tf_idf_vectorizer.get_feature_names())


def data_set_difference():
    reviews_train_info = sklearn.datasets.load_files('reviews/train')
    reviews_dev_info = sklearn.datasets.load_files('reviews/dev')
    train_words = get_feature_set(train_info=reviews_train_info)
    print len(train_words)
    dev_words = get_feature_set(train_info=reviews_dev_info)
    print len(dev_words)
    diff_words = dev_words - train_words
    print len(diff_words)
    for ele in diff_words:
        print ele


if __name__ == '__main__':
    sgd_classifier = sklearn.linear_model.SGDClassifier(loss="log", penalty="elasticnet", n_iter=10)
    compare_feature_extraction(sgd_classifier)
    #data_set_difference()