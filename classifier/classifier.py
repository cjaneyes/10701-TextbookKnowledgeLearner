# !/usr/bin/python 

from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def ReadData(train_file):
    data = load_svmlight_file(train_file)
    return data

def classify(data, clf, n_fold):
    data = ReadData()
    cv = KFold(n=data.shape[0], n_folds=n_fold)
    # Get Data and Label, according to input format

    for train, test in cv:
        clf.fit(X[train], y[train])
        acc = clf.predict(X[test], y[test])
        val += acc
    val /= n_fold
    return val

def Classifier(data):
    gnb = GaussianNB()
    svc = SVC(kernel='linear')
    lclf = [(gnb, 'Gaussian NB')
           ,(svc, 'Support Vector classifier')]
    for clf, name in lclf:
        acc = classify(data, clf, 10)
        print 'Classifier %s has accuracy %d' %(name, acc)

if __name__ == '__main__':
    data = ReadData('./train_file')
    #Classifier(data)
