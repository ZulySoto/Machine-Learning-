import  csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from surprise import SVD
from surprise import KNNBasic
from surprise import accuracy
from surprise import Reader
from surprise import Dataset

from surprise.model_selection import cross_validate
from surprise.model_selection import split
from surprise.model_selection.split import train_test_split
from surprise.model_selection import KFold

def readData(filename):
    file = open(filename)
    reader = csv.reader(file)

    header = next(reader)
    trainData = []

    count = 0
    for x in reader:
        trainData.append(x)
        print(count)
        count = count + 1


    trainData = np.array(trainData)
    header = np.array(header)
    return header, trainData

#How to read data from the file to use the surprise library
def readDataSurprise(filename):
    reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines = 1)
    trainData = Dataset.load_from_file(filename, reader = reader)
    #trainData = trainData.build_full_trainset()
    return trainData

#Do the PMF with cross validation
def PMFUpdate(trainData):
    #split = KFold(n_splits = 5)
    RMSE = 0
    MAE = 0
    count = 0

    #loops five times fro cross validation
    while(count < 5):
        train, test = train_test_split(trainData) #splits the data
        predict = SVD().fit(train).test(test)

        RMSE = RMSE + accuracy.rmse(predict)
        MAE = MAE + accuracy.mae(predict)
        count = count + 1

    RMSE = RMSE/ 5
    MAE = MAE / 5
    return RMSE, MAE

def userBased(trainData):
    RMSE = 0
    MAE = 0
    count = 0
    #loops five times fro cross validation
    while(count < 5):
        train, test = train_test_split(trainData) #splits the data
        predict =KNNBasic().fit(train).test(test)

        RMSE = RMSE + accuracy.rmse(predict)
        MAE = MAE + accuracy.mae(predict)
        count = count + 1

    RMSE = RMSE/ 5
    MAE = MAE / 5
    return RMSE, MAE

def itemBased(trainData):
    RMSE = 0
    MAE = 0
    count = 0
    sim_options = {'user_based': False}
    #loops five times fro cross validation
    while(count < 5):
        train, test = train_test_split(trainData) #splits the data
        predict =KNNBasic(sim_options = sim_options).fit(train).test(test)

        RMSE = RMSE + accuracy.rmse(predict)
        MAE = MAE + accuracy.mae(predict)
        count = count + 1

    RMSE = RMSE/ 5
    MAE = MAE / 5
    return RMSE, MAE

#Experiment with different similarity
def userBasedSimilarities(trainData):
    cr = 0  #Cosine RMSE
    crArray = []
    cm = 0  #Cosine MAE
    cmArray = []

    mr = 0  #MSD RMSE
    mrArray = []
    mm = 0  #MSD MAE
    mmArray = []

    pr = 0  #Pearson RMSE
    prArray = []
    pm = 0  #Pearson MAE
    pmArray = []
    count = 0

    sim_options = {'name' : 'cosine'} #cosine
    sim_options1 = {'name' : 'msd'} #MSD
    sim_options2 = {'name' : 'pearson'} #Pearson

    #loops five times for cross validation
    while(count < 5):
        train, test = train_test_split(trainData) #splits the data

        #Uses Cosine Similarity
        predict = KNNBasic(sim_options = sim_options).fit(train).test(test)
        crArray.append(accuracy.rmse(predict))
        cr = cr + accuracy.rmse(predict)
        cmArray.append(accuracy.mae(predict))
        cm = cm + accuracy.mae(predict)

        #Uses MSD Similarity
        predict = KNNBasic(sim_options = sim_options1).fit(train).test(test)
        mrArray.append(accuracy.rmse(predict))
        mr = mr + accuracy.rmse(predict)
        mmArray.append(accuracy.mae(predict))
        mm = mm + accuracy.mae(predict)

        #Uses Pearson Similarity
        predict = KNNBasic(sim_options = sim_options2).fit(train).test(test)
        prArray.append(accuracy.rmse(predict))
        pr = pr + accuracy.rmse(predict)
        pmArray.append(accuracy.mae(predict))
        pm = pm + accuracy.mae(predict)
        count = count + 1

    cr = cr / 5  #Average Cosine RMSE
    cm = cm / 5 #Average Cosine MAE

    mr = mr / 5  #Average MSD RMSE
    mm = mr / 5  #Average MSD MAE

    pr = pr / 5  #Pearson RMSE
    pm = pm / 5  #Pearson MAE

    return cr, cm, mr, mm, pr, pm, crArray, cmArray, mrArray, mmArray, prArray, prArray

def itemBasedSimilarities(trainData):
    cr = 0  #Cosine RMSE
    crArray = []
    cm = 0  #Cosine MAE
    cmArray = []

    mr = 0  #MSD RMSE
    mrArray = []
    mm = 0  #MSD MAE
    mmArray = []

    pr = 0  #Pearson RMSE
    prArray = []
    pm = 0  #Pearson MAE
    pmArray = []
    count = 0

    sim_options = {'name' : 'cosine', 'user_based': False} #cosine
    sim_options1 = {'name' : 'msd', 'user_based': False} #MSD
    sim_options2 = {'name' : 'pearson', 'user_based': False} #Pearson

    #loops five times for cross validation
    while(count < 5):
        train, test = train_test_split(trainData) #splits the data

        #Uses Cosine Similarity
        predict = KNNBasic(sim_options = sim_options).fit(train).test(test)
        crArray.append(accuracy.rmse(predict))
        cr = cr + accuracy.rmse(predict)
        cmArray.append(accuracy.mae(predict))
        cm = cm + accuracy.mae(predict)

        #Uses MSD Similarity
        predict = KNNBasic(sim_options = sim_options1).fit(train).test(test)
        mrArray.append(accuracy.rmse(predict))
        mr = mr + accuracy.rmse(predict)
        mmArray.append(accuracy.mae(predict))
        mm = mm + accuracy.mae(predict)

        #Uses Pearson Similarity
        predict = KNNBasic(sim_options = sim_options2).fit(train).test(test)
        prArray.append(accuracy.rmse(predict))
        pr = pr + accuracy.rmse(predict)
        pmArray.append(accuracy.mae(predict))
        pm = pm + accuracy.mae(predict)
        count = count + 1

    cr = cr / 5  #Average Cosine RMSE
    cm = cm / 5 #Average Cosine MAE

    mr = mr / 5  #Average MSD RMSE
    mm = mr / 5  #Average MSD MAE

    pr = pr / 5  #Pearson RMSE
    pm = pm / 5  #Pearson MAE

    return cr, cm, mr, mm, pr, pm, crArray, cmArray, mrArray, mmArray, prArray, prArray

#Experiment with different ks
def userBasedK(trainData, k):
    mr = 0
    mm = 0

    count = 0

    #Options for similarity
    sim_options = {'name' : 'msd'}

    #loops five times for cross validation
    while(count < 5):
        train, test = train_test_split(trainData) #splits the data

        #Uses Cosine Similarity
        predict = KNNBasic(k = k, sim_options = sim_options).fit(train).test(test)
        mr = mr + accuracy.rmse(predict)
        mm = mm + accuracy.mae(predict)
        count = count + 1

    mr = mr / 5
    mm = mm / 5

    return mr, mm

trainData = readDataSurprise("./ratings_small.csv")

#Tried to run the ratings.csv however it took 5 hours so it became inpossible to run for the assignment
#trainData = readDataSurprise("./ratings.csv")

#PMF
pmfRSME, pmfMAE = PMFUpdate(trainData)
print("PMF Performance")
print(pmfRSME)
print(pmfMAE)

#User Based
uRSME, uMAE= userBased(trainData)
print("User Based Performace")
print(uRSME)
print(uMAE)

#Item Based
iRSME, iMAE= itemBased(trainData)
print("Item Based Performace")
print(iRSME)
print(iMAE)

#Experiments with different similaritites user based
cr, cm, mr, mm, pr, pm, crArray, cmArray, mrArray, mmArray, prArray, pmArray = userBasedSimilarities(trainData)
x = [1,2,3,4,5]

#Plots the five RSME User Based
plt.figure(figsize=(8,8))
plt.plot(x, crArray, label = "Cosine")
plt.plot(x, mrArray, label = "MSD")
plt.plot(x, prArray, label = "Pearson")
plt.legend(loc = 'upper right')
plt.title("RSME Over 5 Splits with Different Measures of Similarities (User Based)")
plt.savefig("RSMEUserSimilarity.png")

#Plots the five MAE User Based
plt.figure(figsize=(8,8))
plt.plot(x, cmArray, label = "Cosine")
plt.plot(x, mmArray, label = "MSD")
plt.plot(x, pmArray, label = "Pearson")
plt.legend(loc = 'upper right')
plt.title("MAE Over 5 Splits with Different Measures of Similarities (User Based)")
plt.savefig("MAEUserSimilarity.png")

print("-------------------------------------------------------------")
print("Average RMSE")
print("Cosine: ", cr)
print("MSD: ", mr)
print("Pearson: ", pr)
print("Average MAE")
print("Cosine: ", cm)
print("MSD: ", mm)
print("Pearson: ", pm)
print("------------------------------------------------------------")

#Experiments with different similaritites item based
cr, cm, mr, mm, pr, pm, crArray, cmArray, mrArray, mmArray, prArray, pmArray = itemBasedSimilarities(trainData)
Plots the five RSME Item Based
plt.figure(figsize=(8,8))
plt.plot(x, crArray, label = "Cosine")
plt.plot(x, mrArray, label = "MSD")
plt.plot(x, prArray, label = "Pearson")
plt.legend(loc = 'upper right')
plt.title("RSME Over 5 Splits with Different Measures of Similarities (Item Based)")
plt.savefig("RSMEItemSimilarity.png")

#Plots the five MAE Item Based
plt.figure(figsize=(8,8))
plt.plot(x, cmArray, label = "Cosine")
plt.plot(x, mmArray, label = "MSD")
plt.plot(x, pmArray, label = "Pearson")
plt.legend(loc = 'upper right')
plt.title("MAE Over 5 Splits with Different Measures of Similarities (Item Based)")
plt.savefig("MAEItemSimilarity.png")
print("-------------------------------------------------------------")
print("Average RMSE")
print("Cosine: ", cr)
print("MSD: ", mr)
print("Pearson: ", pr)
print("Average MAE")
print("Cosine: ", cm)
print("MSD: ", mm)
print("Pearson: ", pm)
print("------------------------------------------------------------")

#Experiment with different K's
kMR = []
kMM = []
k = 5
while(k <= 60):
    mr, mm = userBasedK(trainData, k)
    kMR.append(mr)
    kMM.append(mm)
    k = k + 5
x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

#Plots RSME Results for K
plt.figure(figsize=(8,8))
plt.plot(x, kMR)
plt.title("RSME for Different Values of K")
plt.savefig("RSMEK.png")

#Plots MAE Results for K
plt.figure(figsize=(8,8))
plt.plot(x, kMM)
plt.title("MAE for Different Values of K")
plt.savefig("MAEK.png")

kMR = []
kMM = []
k = 10
while(k <= 30):
    mr, mm = userBasedK(trainData, k)
    kMR.append(mr)
    kMM.append(mm)
    k = k + 1
x = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

#Plots RSME Results for K
plt.figure(figsize=(8,8))
plt.plot(x, kMR)
plt.title("RSME for Different Values of K")
plt.savefig("RSMEK1030.png")

#Plots MAE Results for K
plt.figure(figsize=(8,8))
plt.plot(x, kMM)
plt.title("MAE for Different Values of K")
plt.savefig("MAEK1030.png")
