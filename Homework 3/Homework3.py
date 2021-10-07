import csv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Read data from csv file
def readData():
    titanicFile = open('train.csv')
    titanicReader = csv.reader(titanicFile)
    header = next(titanicReader)
    trainTitanic = []
    for x in titanicReader:
        trainTitanic.append(x)


    #print(header)
    return trainTitanic, header
#Splits the data into the features and labels
def splitData(trainData, rowTotal):
    row = 0
    y_train = []
    x_train = []
    while(row < rowTotal):
        y_train.append(trainData[row][1])
        x_train.append(trainData[row][2:9])
        row = row + 1

    x,x_vali, y, y_vali = train_test_split(x_train, y_train)

    return x, y, x_vali, y_vali

def linear(x_train, y_train):
    sum = 0
    #Type casts to int
    xx_train = np.array(x_train)
    yy_train = np.array(y_train)
    linearSVM = svm.SVC(kernel = 'linear')


    k = KFold(n_splits = 5)
    for xIndex, yIndex in k.split(x_train):
        x, x_vali = xx_train[xIndex], xx_train[yIndex]
        y, y_vali = yy_train[xIndex], yy_train[yIndex]
        #trains the Decision tree
        linearSVM.fit(x, y)
        #Predict
        predictions = linearSVM.predict(x_vali)
        #calculates the accuracy
        accuracy = metrics.accuracy_score(y_vali, predictions)
        sum = sum + accuracy

        #takes the average
    average = sum/5
    return average

def polynomial(x_train, y_train):
    sum = 0
    #Type casts to int
    xx_train = np.array(x_train)
    yy_train = np.array(y_train)
    linearSVM = svm.SVC(kernel = 'poly')


    k = KFold(n_splits = 5)
    for xIndex, yIndex in k.split(x_train):
        x, x_vali = xx_train[xIndex], xx_train[yIndex]
        y, y_vali = yy_train[xIndex], yy_train[yIndex]
        #trains the Decision tree
        linearSVM.fit(x, y)
        #Predict
        predictions = linearSVM.predict(x_vali)
        #calculates the accuracy
        accuracy = metrics.accuracy_score(y_vali, predictions)
        sum = sum + accuracy

        #takes the average
    average = sum/5
    return average
trainData, header = readData()
rowTotal = len(trainData)

def RBF(x_train, y_train):
    sum = 0
    #Type casts to int
    xx_train = np.array(x_train)
    yy_train = np.array(y_train)
    linearSVM = svm.SVC(kernel = 'rbf')


    k = KFold(n_splits = 5)
    for xIndex, yIndex in k.split(x_train):
        x, x_vali = xx_train[xIndex], xx_train[yIndex]
        y, y_vali = yy_train[xIndex], yy_train[yIndex]
        #trains the Decision tree
        linearSVM.fit(x, y)
        #Predict
        predictions = linearSVM.predict(x_vali)
        #calculates the accuracy
        accuracy = metrics.accuracy_score(y_vali, predictions)
        sum = sum + accuracy

        #takes the average
    average = sum/5
    return average
x_train, y_train, x_vali, y_vali  = splitData(trainData, rowTotal)

linearMetrics = linear(x_train, y_train)
polyMetrics = polynomial(x_train, y_train)
rbfMetrics = RBF(x_train, y_train)

print(linearMetrics)
print(polyMetrics)
print(rbfMetrics)
