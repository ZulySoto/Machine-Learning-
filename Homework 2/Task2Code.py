import csv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from dtreeviz.trees import dtreeviz

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

# Creates decision Tree
def decisionTree(x_train, y_train):
    sum = 0
    #Type casts to int
    xx_train = np.array(x_train)
    yy_train = np.array(y_train)

    #creates decisionTree
    dTree = DecisionTreeClassifier(splitter = "random")

    #Does the 5 fold validation possible
    k = KFold(n_splits = 5)
    for xIndex, yIndex in k.split(x_train):
        x, x_vali = xx_train[xIndex], xx_train[yIndex]
        y, y_vali = yy_train[xIndex], yy_train[yIndex]
        #trains the Decision tree
        dTree.fit(x, y)
        #Predict
        predictions = dTree.predict(x_vali)
        #calculates the accuracy
        accuracy = metrics.accuracy_score(y_vali, predictions)
        sum = sum + accuracy

    #takes the average
    average = sum/5
    return average, dTree
#Creates random forest tree
def randomForestTree(x_train, y_train):
    sum = 0
    #Type casts to int
    xx_train = np.array(x_train)
    yy_train = np.array(y_train)
    #creates random forest
    randTree = RandomForestClassifier()

    #Does the 5 fold validation possible
    k = KFold(n_splits = 5)
    for xIndex, yIndex in k.split(x_train):
        x, x_vali = xx_train[xIndex], xx_train[yIndex]
        y, y_vali = yy_train[xIndex], yy_train[yIndex]
        #trains the Decision tree
        randTree.fit(x, y)
        #Predict
        predictions = randTree.predict(x_vali)
        accuracy = metrics.accuracy_score(y_vali, predictions)
        sum = sum + accuracy

    average = sum/5
    return average

#Prints out the decision tree
def plotTree(dTree):
    #Plots a decision tree, uncomment to do so
    tree.plot_tree(dTree, fontsize = 3, impurity = False, label = 'root', feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch"], precision = 1, filled = True)
    plt.show()

def fancyPlotTree(dTree, x_train, y_train):
    graph = dtreeviz(dTree, x_train, y_train, feature_names = ["Pclass", "Sex", "Age", "SibSp", "Parch"])
    graph.view()
trainData, header = readData()
rowTotal = len(trainData)

x_train, y_train, x_vali, y_vali  = splitData(trainData, rowTotal)


decisionTreeAverage, dTree = decisionTree(x_train, y_train)
print(decisionTreeAverage)

randTreeAverage= randomForestTree(x_train, y_train)
print(randTreeAverage)

#Plots the decision tree
plotTree(dTree)
