# !/usr/bin/python 

import scipy
from sklearn.datasets import load_svmlight_file
from sklearn.cross_validation import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

class Classifier:
    # __init__
    def __init__(self,train_file):
        self.train_file = train_file
        self.data = self.readData(train_file)
        self.X = self.data[0]
        self.y = self.data[1]
        #self.preprocess()

    '''
        read data from file
    '''
    def readData(self,train_file):
        data = load_svmlight_file(train_file)
        return data

    '''
        normalize or rescale the data
    '''    
    def preprocess(self):
        self.X = preprocessing.normalize(self.X,axis=1)


    '''
        provide the classification of a single classifier
    '''
    def classify(self,X, y, clf, metrics, n_fold):

        cv = KFold(n=X.shape[0], n_folds=n_fold)
        val_p = 0 
        val_r = 0
        val_f = 0
        for train, test in cv:
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            if metrics=="accuracy":
                val +=  accuracy_score(y[test], y_pred)
            elif metrics =="all":
                val_p += precision_score(y[test], y_pred,average='binary') #  binary means only pos_label
                val_r += recall_score(y[test], y_pred,average='binary')
                val_f += f1_score(y[test], y_pred,average='binary')
        if metrics == "all":
            return [val_p/n_fold, val_r/n_fold, val_f/n_fold]
        elif metrics == "accuracy":
            return val/n_fold
    '''
        This part provides the classification models.
    '''
    def svmLearn(self,kernel_f,metrics):
        # self.X  and self.y
        svc = SVC(kernel=kernel_f)
        lclf = [(svc, 'Support Vector classifier')]
        for clf, name in lclf:
            prec = self.classify(self.X.toarray(), self.y, clf, metrics, 5)

            if metrics == "accuracy":
                print 'Classifier %s has %s %.4f' %(name, metrics, values)
            else:
                self.outputMetrics(prec,name)

    def GaussianLearn(self,metrics):
        gnb = GaussianNB()
        lclf = [(gnb, 'Gaussian NB')]
        for clf, name in lclf:
            prec = self.classify(self.X.toarray(), self.y, clf, metrics, 5)
            if metrics == "accuracy":
                print 'Classifier %s has %s %.4f' %(name, metrics, values)
            else:
                self.outputMetrics(prec,name)
    '''
        Model part ends!
    '''       

    def outputMetrics(self,values,name):
            metrics = ["precision", "recall", "f1-score"]
            for i in range(len(values)):
                print 'Classifier %s has %s %.4f' %(name, metrics[i], values[i])

if __name__ == '__main__':
    c = Classifier('./features.txt')
    c.svmLearn("linear","all")
    #c.GaussianLearn("all")
