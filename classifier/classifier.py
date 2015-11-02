# !/usr/bin/python 

from sklearn.datasets import load_svmlight_file

def ReadData():
    data = load_svmlight_file("")
    return data

ReadData()
