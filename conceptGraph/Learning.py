import scipy
import numpy
import random
from sklearn.linear_model import LogisticRegression

class Learner():
	def __init__(self, file, train_file, test_file):
		self.clf = LogisticRegression()
        self.train_file = train_file
        self.test_file = test_file
        self.Data = load_svmlight_file(file)
        self.train_X = self.read_data(train_file, n_features=self.Data[0].shape[1])[0]
        self.train_y = self.read_data(train_file, n_features=self.Data[0].shape[1])[1]
        self.test_X = self.read_data(test_file, n_features=self.Data[0].shape[1])[0]
        self.test_y = self.read_data(test_file, n_features=self.Data[0].shape[1])[1]
        assert len(self.lines)==(self.Data[0].shape[0])

    '''
        read data from file
    '''
    @staticmethod
    def read_data(train_file, n_features):
        data = load_svmlight_file(train_file, n_features=n_features)
        print data[0]
        print data[1]
        return data

    def classify(self):

        val_a = 0
        #val_p = 0 
        #val_r = 0
        #val_f = 0

        clf.fit(self.train_X, self.train_y)
        y_pred = clf.predict(self.test_X)

        # calculate evaluation metrics: [accuracy, precision, recall, f1_score]
        val_a = accuracy_score(self.test_y,  y_pred)
        #val_p = precision_score(y_test, y_pred, average='binary')
        #val_r = recall_score(y_test, y_pred, average='binary')
        #val_f = f1_score(y_test, y_pred, average='binary')

        return val_a


