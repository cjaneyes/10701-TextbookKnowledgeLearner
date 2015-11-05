# !/usr/bin/python 

import scipy
import numpy
import random
from sklearn.datasets import load_svmlight_file
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.grid_search import ParameterGrid
from sklearn.feature_extraction.text import TfidfTransformer

class Classifier:
    # __init__
    def __init__(self, file, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        
        self.Data = load_svmlight_file(file)

        self.read_lines(file) 
      
        if train_file == "" or test_file == "":
            return
        self.data = self.read_data(train_file, n_features=self.Data[0].shape[1])
        self.X = self.data[0].toarray()
        self.y = self.data[1]

        self.test = self.read_data(test_file, n_features=self.Data[0].shape[1])
        self.X_test = self.test[0].toarray()
        self.y_test = self.test[1]

        self.n_split = 5
        self.n_fold = 10
        assert len(self.lines)==(self.X.shape[0])

    '''
        read data from file
    '''
    @staticmethod
    def read_data(train_file, n_features):
        data = load_svmlight_file(train_file, n_features=n_features)
        return data

    def read_lines(self,train_file):
        self.lines = open(train_file,"r").readlines()
        return self.lines

    '''
        provide sampling method, where:
        sampling = (0) no sampling  (1) undersampling  (2) oversampling
        ratio = expected #pos / #neg
    '''
    @staticmethod
    def sampling(X, y, sampling, ratio):

        lneg = []
        lpos = []
        for i in range(len(y)):
            if y[i] == 1:
                lpos.append(i)
            else:
                lneg.append(i)

        print y
        if sampling == 1:
            num = int(len(lpos)/ratio)
            ln = random.sample(lneg, num)
            X_s = numpy.empty((0,X.shape[1]), float)
            y_s = numpy.empty((0,), float)
            for i in lpos:
                X_s = numpy.append(X_s, numpy.array([X[i]]), axis=0)
                y_s = numpy.append(y_s, numpy.array([y[i]]), axis=0)
            for i in ln:
                X_s = numpy.append(X_s, numpy.array([X[i]]), axis=0)
                y_s = numpy.append(y_s, numpy.array([y[i]]), axis=0)
        elif sampling == 2:
            X_s = X
            y_s = y
            num = int((len(lneg)*ratio/len(lpos)))-1
            for i in range(len(y)):
                if y[i] == 1:
                    for j in range(num):
                        X_s = numpy.append(X_s, numpy.array([X[i]]), axis=0)
                        y_s = numpy.append(y_s, numpy.array([y[i]]), axis=0)

        return (X_s, y_s)


    '''
        provide the classification of a single Classifier
        training and testing with cross validation
    '''
    @staticmethod
    def cv(X, y, clf, n_fold):

        cv = cross_validation.StratifiedKFold(y, n_folds=n_fold)
        val_a = 0
        val_p = 0 
        val_r = 0
        val_f = 0
        for train, test in cv:
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])

            # calculate evaluation metrics: [accuracy, precision, recall, f1_score]

            val_a += accuracy_score(y_test, y_pred)
            val_p += precision_score(y_test, y_pred, average='binary')
            val_r += recall_score(y_test, y_pred, average='binary')
            val_f += f1_score(y_test, y_pred, average='binary')

        return [val_a/n_fold, val_p/n_fold, val_r/n_fold, val_f/n_fold]


    '''
        provide the classification of a single Classifier
        trainging with input parameters and testing with  unseen testing data
    '''
    @staticmethod
    def classify(X, y, X_test, y_test, clf):

        val_a = 0
        val_p = 0 
        val_r = 0
        val_f = 0

        clf.fit(X, y)
        y_pred = clf.predict(X_test)

        # calculate evaluation metrics: [accuracy, precision, recall, f1_score]

        val_a = accuracy_score(y_test,  y_pred)
        val_p = precision_score(y_test, y_pred, average='binary')
        val_r = recall_score(y_test, y_pred, average='binary')
        val_f = f1_score(y_test, y_pred, average='binary')

        return [val_a, val_p, val_r, val_f]



    '''
        preprocessing the input data, where:
        sampling = (0) no sampling  (1) undersampling  (2) oversampling
        ratio = expected #pos / #neg
    '''    
    def preprocess(self, sampling, ratio):
        tfidf = TfidfTransformer()
        tfidf.fit_transform(self.X)
        tfidf.transform(self.X_test)
        
        if sampling != 0:
            self.X_train, self.y_train = self.sampling(self.X, self.y, sampling, ratio)
        #self.X = preprocessing.normalize(self.X,axis=1)


    '''
        This part provides the classification models.
    '''
    def svm(self, c, kernel_f, g, cv):
        svc = SVC(C=c, kernel=kernel_f, gamma=g)
        name  = 'Support Vector Machine with kernel: %s C:%d gamma:%f' %(kernel_f, c, g)
        if cv == True:
            res = self.cv(self.X, self.y, svc, self.n_fold)
        else:
            res = self.classify(self.X, self.y, self.X_test, self.y_test, svc)
        self.write_metrics(res, name)

    def gaussian_nb(self, cv):
        gnb = GaussianNB()
        name = 'Gaussian NB'
        
        if cv == True:
            res = self.cv(self.X, self.y, gnb, self.n_fold)
        else:
            res = self.classify(self.X, self.y, self.X_test, self.y_test, gnb)
        self.write_metrics(res, name)
    
    def gradient_boosting(self, cv):
        gbc = GradientBoostingClassifier()
        name = 'GradientBoostingClassifier'

        if cv == True:
            res = self.cv(self.X, self.y, gbc, self.n_fold)
        else:
            res = self.classify(self.X, self.y, self.X_test, self.y_test, gbc)
        self.write_metrics(res, name)

    def decision_tree(self, cv):
        dt = DecisionTreeClassifier(random_state=0)
        name = 'Decision Tree'

        if cv == True:
            res = self.cv(self.X, self.y, dt, self.n_fold)
        else:
            res = self.classify(self.X, self.y, self.X_test, self.y_test, dt)
        self.write_metrics(res, name)


    def write_metrics(self,values,name):
            metrics = ["accuracy", "precision", "recall", "f1-score"]
            for i in range(len(values)):
                print 'Classifier %s has %s %.4f' %(name, metrics[i], values[i])
            print '\n'

    '''
        Split input data into training set and testing set
    '''
    def split(self, n_split):
        cv = cross_validation.StratifiedKFold(self.Data[1], n_folds=n_split)
        train = []
        test = []
        for train, test in cv:
            break
        
        write_file = open("sen/sen.train","w")
        for itr in train:
            write_file.write(self.lines[itr])
        write_file = open("sen/sen.test","w")
        for itr in test:
            write_file.write(self.lines[itr])
        

if __name__ == '__main__':
    #c1 = Classifier('./bow/feature.bow')
    c = Classifier('./sen/feature.bow.sen', "", "")
    c.split(5)
    #c.svm(1, 'linear', 0, False)

    #crange = [1,10,100]
    #grange = numpy.linspace(0, 0.1, num=20)
    #for C in crange:
    #    c.svm(C, 'linear', 0, True)
    #for C in crange:
    #    for g in grange:
    #        c.svm(C, 'rbf', g, True)

    #c.gaussian_nb()
    #c.gradient_boosting()
