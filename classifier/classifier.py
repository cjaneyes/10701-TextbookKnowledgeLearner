# !/usr/bin/python 

import scipy
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

'''
    read data from file
'''
def ReadData(train_file):
    data = load_svmlight_file(train_file)
    return data

'''
    provide the classification of a single classifier
'''
def classify(X, y, clf, n_fold):

    cv = KFold(n=X.shape[0], n_folds=n_fold)
    val = 0
    for train, test in cv:
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])
        val +=  accuracy_score(y[test], y_pred)
    val /= n_fold
    return val

'''
    provide the classification main process
'''
def Classifier(data):
    X = data[0]
    y = data[1]
    gnb = GaussianNB()
    svc = SVC(kernel='linear')
    lclf = [(gnb, 'Gaussian NB')]
           #,(svc, 'Support Vector classifier')]
    for clf, name in lclf:
        prec = classify(X.toarray(), y, clf, 5)
        print 'Classifier %s has accuracy %.4f' %(name, prec)


if __name__ == '__main__':
    data = ReadData('./features.txt')
    Classifier(data)
