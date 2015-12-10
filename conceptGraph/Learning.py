import scipy
import numpy
import random
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class Learner():
    def __init__(self, file, train_file, test_file, label_file, feature_train_file, feature_test_file):
        self.init(file, train_file, test_file, label_file, feature_train_file, feature_test_file)

    def init(self, file, train_file, test_file, label_file, feature_train_file, feature_test_file):
        self.clf = LogisticRegression()
        self.label = self.read_label(label_file)
        self.genLibSVM(feature_train_file, train_file)
        self.genLibSVM(feature_test_file, test_file)

        self.train_file = train_file
        self.test_file = test_file
        self.Data = load_svmlight_file(file)
        self.train_X = self.read_data(train_file, n_features=self.Data[0].shape[1])[0]
        self.train_y = self.read_data(train_file, n_features=self.Data[0].shape[1])[1]
        self.test_X = self.read_data(test_file, n_features=self.Data[0].shape[1])[0]
        self.test_y = self.read_data(test_file, n_features=self.Data[0].shape[1])[1]
        #assert len(self.lines)==(self.Data[0].shape[0])

    @staticmethod
    def read_label(label_file):
        lines = open(label_file, 'r').readlines()
        dictionary = {}
        #cur_sentence = "start"
        for i in range(0, len(lines), 2):
            sentence = lines[i].strip()
            label_info = lines[i+1]
            parse = label_info.strip().split("\t")
            if len(parse) <= 2:
                sample = parse[0].strip()
                label = 0
            else:
                label = int(parse[0].strip())
                sample = parse[1].strip()
            dictionary[(sentence, sample)] = label
        return dictionary

    '''
        read data from file
    '''
    @staticmethod
    def read_data(train_file, n_features):
        data = load_svmlight_file(train_file, n_features=n_features)
        #print data[0]
        #print data[1]
        return data

    def genLibSVM(self, inputFilePath, outputFilePath):
        lines = open(inputFilePath, 'r').readlines()
        output = open(outputFilePath, 'w')
        for i in range(0, len(lines), 2):
            sentence = lines[i].strip()
            label_info = lines[i+1]
            parse = label_info.strip().split("\t")
            sample = parse[0].strip()
            feature = parse[1].strip()
            label = self.label[(sentence, sample)]
            output.write(str(label)+"\t"+feature+"\n")
        output.close()
            

    def classify(self):

        val_a = 0
        #val_p = 0 
        #val_r = 0
        #val_f = 0

        self.clf.fit(self.train_X, self.train_y)
        y_pred = self.clf.predict(self.test_X)
        cnt = 0
        for item in y_pred:
            if item == 1:
                cnt += 1
        print cnt
        print len(y_pred)
        # calculate evaluation metrics: [accuracy, precision, recall, f1_score]
        val_a = accuracy_score(self.test_y,  y_pred)
        val_p = precision_score(self.test_y, y_pred, average='binary')
        val_r = recall_score(self.test_y, y_pred, average='binary')
        val_f = f1_score(self.test_y, y_pred, average='binary')
        print "accuracy is "+ str(val_a)
        print "precision is " + str(val_p)
        print "recall is " + str(val_r)
        print "F-measure is " + str(val_f)

if __name__ == '__main__':
    #learner = Learner("./Predict/train_uni", "./Predict/uni.sub.literal", "./Predict//uni.feat.literal")
    #learner.classify()
    learner = Learner("./Predict/train_bi", "./Predict/train_bi", \
        "./Predict/train_bi", "./Predict/bi.sub.literal", \
        "./Predict/bi.feat.literal.prune","./Predict/bi.test.literal")
    learner.classify()





