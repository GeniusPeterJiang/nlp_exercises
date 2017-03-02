import newsreader
import classifier
import scipy
import numpy

class DataLoader(object) :
    '''Class for managing the mapping from newsgroup-style classification
    problem text files in the file system to feature matrix representations
    as used for classification'''

    def __init__(self, train_dir, test_dir,
                 train_dir_pos, train_dir_neg,
                 test_dir_pos, test_dir_neg,
                 vectorizer,
                 reduction=None):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.initialized = False
        self.train_dir_pos = train_dir_pos
        self.train_dir_neg = train_dir_neg
        self.test_dir_pos = test_dir_pos
        self.test_dir_neg = test_dir_neg
        self.vectorizer = vectorizer
        self.features = None
        self.reduction = reduction

    def initialize(self):
        '''intialize: query the file system to get the instances
        to work with and use the passed callback to compile the
        features to classify with'''
        self.train_info = classifier.read_train_data(self.train_dir, self.train_dir_pos, self.train_dir_neg)
        classifier.remove_failed_documents(self.train_info)
        self.test_info = classifier.read_train_data(self.test_dir, self.test_dir_pos, self.test_dir_neg)
        classifier.remove_failed_documents(self.test_info)
        self.vectorizer.fit(self.train_info.data)
        self.features = self.vectorizer.get_feature_names()
        self.initialized = True

    def _info_to_features(self, info):
        '''helper function for compiling a feature matrix from a list of files.
        each row in the matrix corresponds to one of the files.
        the matrix is returned in sparse CSR representation.
        the passed callback is used to get the feature representation
        from the file tokens'''
        x_train = self.vectorizer.transform(info.data)
        if self.reduction:
            x_train = self.reduction(x_train)
        return x_train

    def _get_data(self, info) :
        '''helper function for returning data from a particular class.
        this is where we shuffle the data (if requested)
        to avoid glitches from (e.g.) the order of training elements.'''
        return self._info_to_features(info), info.target


    def training_data(self) :
        '''get matrices for the training data and training categories from the collection'''
        if self.initialized :
            return self._get_data(self.train_info)
        else:
            raise newsreader.DataError("Must call initialize() on Data Manager before accessing data")

    def dev_data(self) :
        '''get matrices for the development data and development categories from the collection'''
        if self.initialized :
            (data_x, data_y) = self._get_data(self.test_info)
            dev_len = len(data_y) / 4
            return data_x[:dev_len], data_y[:dev_len]

        else:
            raise newsreader.DataError("Must call initialize() on Data Manager before accessing data")

    def test_data(self):
        '''get matrices for the test data and test categories from the collection'''
        if self.initialized :
            (data_x, data_y) = self._get_data(self.test_info)
            dev_len = len(data_y) / 4
            return data_x[dev_len:], data_y[dev_len:]
        else:
            raise newsreader.DataError("Must call initialize() on Data Manager before accessing data")

    def centroid_distance(self):
        x, y = self.test_data()
        pos, neg = self.divide_data_x_according_to_y(x, y)
        positive_mean = pos.mean(axis=0)
        negative_mean = neg.mean(axis=0)
        return 1 - scipy.spatial.distance.cosine(positive_mean, negative_mean)

    def divide_data_x_according_to_y(self, data_x, data_y):
        positive = numpy.where(data_y == 0)[0]
        negative = numpy.where(data_y == 1)[0]
        return data_x[positive, :], data_x[negative, :]



