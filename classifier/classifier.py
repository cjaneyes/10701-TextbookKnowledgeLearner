# !/usr/bin/python 

import scipy
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.grid_search import ParameterGrid

class Classifier:
    # __init__
    def __init__(self,train_file):
        self.train_file = train_file
        self.data = self.read_data(train_file)
        self.X = self.data[0]
        self.y = self.data[1]
        self.n_fold = 2

    '''
        read data from file
    '''
    @staticmethod
    def read_data(train_file):
        data = load_svmlight_file(train_file)
        return data

    '''
        preprocessing the input data
    '''    
    def preprocess(self):
        self.X = preprocessing.normalize(self.X,axis=1)


    '''
        provide the classification of a single classifier
    '''
    @staticmethod
    def classify(X, y, clf, n_fold):

        cv = cross_validation.StratifiedKFold(y, n_folds=n_fold)
        val_a = 0
        val_p = 0 
        val_r = 0
        val_f = 0
        for train, test in cv:
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            # calculate evaluation metrics: [accuracy, precision, recall, f1_score]
            #cnt = 0
            #for i in range(len(y_pred)):
            #    if y_pred[i] == y[test][i] and y_pred[i] == 1:
            #        cnt += 1
            #print cnt

            val_a += accuracy_score(y[test], y_pred)
            val_p += precision_score(y[test], y_pred, average='binary')
            val_r += recall_score(y[test], y_pred, average='binary')
            val_f += f1_score(y[test], y_pred, average='binary')

        return [val_a/n_fold, val_p/n_fold, val_r/n_fold, val_f/n_fold]


    '''
        This part provides the classification models.
    '''
    def svm(self, kernel_f):
        svc = SVC(kernel=kernel_f)
        name  = 'Support Vector Machine'
        
        res = self.classify(self.X.toarray(), self.y, svc, self.n_fold)
        self.write_metrics(res, name)

    def gaussian_nb(self):
        gnb = GaussianNB()
        name = 'Gaussian NB'
        
        res = self.classify(self.X.toarray(), self.y, gnb, self.n_fold)
        self.write_metrics(res, name)


    '''
        Model part ends!
    '''       

    def write_metrics(self,values,name):
            metrics = ["accuracy", "precision", "recall", "f1-score"]
            for i in range(len(values)):
                print 'Classifier %s has %s %.4f' %(name, metrics[i], values[i])

if __name__ == '__main__':
    c = Classifier('./features.txt')
    c.preprocess()
    c.svm("rbf")
    #c.GaussianLearn("all")

