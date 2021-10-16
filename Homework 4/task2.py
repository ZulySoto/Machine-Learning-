import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statistics import mean
from random import randint

# Read data from csv file
def readData(filename):
    file = open(filename)
    reader = pd.read_csv(file)

    trainData = []
    data = []
    x = 1
    y = 1
    while(x < 1000):
        while(y < 784):
            data.append(reader.values[x - 1][y - 1])
            y = y + 1
        trainData.append(data)
        data = []
        y = 1
        x = x + 1

    return trainData

#Does the KMeans algorithm
def kMeansAlgorithm(numOfClusters, trainData, centerClusters, typeOfDistance):

    stopSignal = 1
    counter = 0

    while(stopSignal != 0):
        previousCenterClusters = centerClusters
        print("Interation")
        print(counter)
        clustersWithData = setDatatoCentroid(numOfClusters, centerClusters, trainData, typeOfDistance)

        centerClusters = newCenterClusters(clustersWithData)
        counter  =  counter + 1

        #Checks if the clusters have changed
        stop = np.subtract(previousCenterClusters, centerClusters)
        if(np.all(stop == 0) or counter > 50):
           stopSignal = 0

    return clustersWithData, centerClusters

#Gets the new centers for the clusters
def newCenterClusters(clustersWithData):
    newCenter1 = np.empty(clusterLength1)
    newCenter2 = np.empty(clusterLength1)
    newCenter3 = np.empty(clusterLength1)
    newCenter4 = np.empty(clusterLength1)
    newCenter5 = np.empty(clusterLength1)
    newCenter6 = np.empty(clusterLength1)
    newCenter7 = np.empty(clusterLength1)
    newCenter8 = np.empty(clusterLength1)
    newCenter9 = np.empty(clusterLength1)
    newCenter10 = np.empty(clusterLength1)

    clusterLength1 = len(clustersWithData[0])
    clusterLength2 = len(clustersWithData[1])
    clusterLength3 = len(clustersWithData[2])
    clusterLength4 = len(clustersWithData[3])
    clusterLength5 = len(clustersWithData[4])
    clusterLength6 = len(clustersWithData[5])
    clusterLength7 = len(clustersWithData[6])
    clusterLength8 = len(clustersWithData[7])
    clusterLength9 = len(clustersWithData[8])
    clusterLength10 = len(clustersWithData[9])


    #Cluster 1
    x = 0
    y = 0
    addition = np.empty(clusterLength1)
    while(x < clusterLength1):
        while(y < clusterLength1):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter1[z] = additon[z]
        z = z + 1

    #Cluster 2
    x = 0
    y = 0
    addition = np.empty(clusterLength2)
    while(x < clusterLength2):
        while(y < clusterLength2):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter2[z] = additon[z]
        z = z + 1

    #Cluster 3
    x = 0
    y = 0
    addition = np.empty(clusterLength3)
    while(x < clusterLength3):
        while(y < clusterLength3):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter3[z] = additon[z]
        z = z + 1


    #Cluster 4
    x = 0
    y = 0
    addition = np.empty(clusterLength4)
    while(x < clusterLength4):
        while(y < clusterLength4):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter4[z] = additon[z]
        z = z + 1

    #Cluster 5
    x = 0
    y = 0
    addition = np.empty(clusterLength5)
    while(x < clusterLength5):
        while(y < clusterLength5):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter5[z] = additon[z]
        z = z + 1

    #Cluster 6
    x = 0
    y = 0
    addition = np.empty(clusterLength6)
    while(x < clusterLength6):
        while(y < clusterLength6):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter6[z] = additon[z]
        z = z + 1

    #Cluster 7
    x = 0
    y = 0
    addition = np.empty(clusterLength7)
    while(x < clusterLength7):
        while(y < clusterLength7):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter7[z] = additon[z]
        z = z + 1

    #Cluster 8
    x = 0
    y = 0
    addition = np.empty(clusterLength8)
    while(x < clusterLength8):
        while(y < clusterLength8):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter8[z] = additon[z]
        z = z + 1

    #Cluster 9
    x = 0
    y = 0
    addition = np.empty(clusterLength9)
    while(x < clusterLength9):
        while(y < clusterLength9):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter9[z] = additon[z]
        z = z + 1

    #Cluster 10
    x = 0
    y = 0
    addition = np.empty(clusterLength1)
    while(x < clusterLength10):
        while(y < clusterLength10):
            addition[x][y] = addition[x][y] + clustersWithData[x][y]
            y = y + 1
        x = x + 1

    while(z < len(addition)):
        newCenter10[z] = additon[z]
        z = z + 1

    centerClusters = [newCenter1, newCenter2, newCenter3, newCenter4, nerCenter5, newCenter6, newCenter7, newCenter8, newCenter9, newCenter10]

    return centerClusters

def setDatatoCentroid(numOfClusters, centerClusters, trainData, typeOfDistance):

    #Eucledian Distance
    if(typeOfDistance == 1):
        clustersWithData = eucledianDistance(trainData, centerClusters, numOfClusters)

    # Cosine Similarity
    if(typeOfDistance == 2):
        clustersWithData = cosineSimilarity(trainData, centerClusters, numOfClusters)

    # Generalized Jarcard Similarity
    if(typeOfDistance == 3):
        clustersWithData = jaccard(trainData, centerClusters, numOfClusters)

    return clustersWithData

#Eucledian Distance
def eucledianDistance(trainData, centerClusters, numOfClusters):
    print("Eucledian Distance")
    clusters1 = []
    clusters2 = []
    clusters3 = []
    clusters4 = []
    clusters5 = []
    clusters6 = []
    clusters7 = []
    clusters8 = []
    clusters9 = []
    clusters10 = []

    counter = 0
    z = 0
    #X should be the rows
    for x in trainData:
        squareRoot = np.empty(numOfClusters)
        while(z < numOfClusters):
            y = 0
            add = 0
            while(y < 783):
                subtract = np.subtract(trainData[x][y], centerClusters[0][0][y])
                square = subtract * subtract
                add = add + square
                y = y + 1
            add = 0
            if(z < numOfClusters):
                squareRoot[z] = np.sqrt(add)
            z = z + 1

        minIndex = np.argmin(squareRoot)

        if(minIndex == 0):
            clusters1.append(trainData[x])

        if(minIndex == 1):
            clusters2.append(trainData[x])

        if(minIndex == 2):
            clusters3.append(trainData[x])

        if(minIndex == 3):
            clusters4.append(trainData[x])

        if(minIndex == 4):
            clusters5.append(trainData[x])

        if(minIndex == 5):
            clusters6.append(trainData[x])

        if(minIndex == 6):
            clusters7.append(trainData[x])

        if(minIndex == 7):
            clusters8.append(trainData[x])

        if(minIndex == 8):
            clusters9.append(trainData[x])

        if(minIndex == 9):
            clusters10.append(trainData[x])

        z = 0
        print(counter)
        counter = counter + 1

    clusterdata = []
    clustersWithData.append(clusters1)
    clustersWithData.append(clusters2)
    clustersWithData.append(clusters3)
    clustersWithData.append(clusters4)
    clustersWithData.append(clusters5)
    clustersWithData.append(clusters6)
    clustersWithData.append(clusters7)
    clustersWithData.append(clusters8)
    clustersWithData.append(clusters9)
    clustersWithData.append(clusters10)

    return clustersWithData

#Cosine Similarity
def cosineSimilarity(trainData, centerClusters, numOfClusters):
    print("Cosine Similarity")
    clusters1 = []
    clusters2 = []
    clusters3 = []
    clusters4 = []
    clusters5 = []
    clusters6 = []
    clusters7 = []
    clusters8 = []
    clusters9 = []
    clusters10 =[]

    z = 0
    #X should be the rows
    for x in trainData:
        simArray = np.empty(numOfClusters, dtype = float)
        while(z < numOfClusters):
            y = 0
            addT = 0
            addC = 0
            product = 0
            while(y < 783):
                product = product + trainData[0][y] * centerClusters[z][0][y]
                multT = trainData[0][y]* trainData[0][y]
                addT  = addT + multT

                multC = centerClusters[z][0][y] * centerClusters[z][0][y]
                addC  = addC + multC
                y = y + 1
            z = z + 1
            squareRootT = np.sqrt(addT)
            squareRootC = np.sqrt(addC)

            denom = squareRootT * squareRootC
            similarity = (product/denom)
            if(z < numOfClusters):
                simArray[z] = similarity

        minIndex = np.argmin(simArray)

        if(minIndex == 0):
            clusters1.append(trainData[x])

        if(minIndex == 1):
            clusters2.append(trainData[x])

        if(minIndex == 2):
            clusters3.append(trainData[x])

        if(minIndex == 3):
            clusters4.append(trainData[x])

        if(minIndex == 4):
            clusters5.append(trainData[x])

        if(minIndex == 5):
            clusters6.append(trainData[x])

        if(minIndex == 6):
            clusters7.append(trainData[x])

        if(minIndex == 7):
            clusters8.append(trainData[x])

        if(minIndex == 8):
            clusters9.append(trainData[x])

        if(minIndex == 9):
            clusters10.append(trainData[x])

        z = 0

    clustersWithData = []
    clustersWithData.append(clusters1)
    clustersWithData.append(clusters2)
    clustersWithData.append(clusters3)
    clustersWithData.append(clusters4)
    clustersWithData.append(clusters5)
    clustersWithData.append(clusters6)
    clustersWithData.append(clusters7)
    clustersWithData.append(clusters8)
    clustersWithData.append(clusters9)
    clustersWithData.append(clusters10)

    return  clustersWithData

def jaccard(trainData, centerClusters, numOfClusters):
    print("Jaccard Similarity")
    clusters1 = []
    clusters2 = []
    clusters3 = []
    clusters4 = []
    clusters5 = []
    clusters6 = []
    clusters7 = []
    clusters8 = []
    clusters9 = []
    clusters10 =[]

    z = 0
    #X should be the rows
    for x in trainData:
        jacarArray = np.empty(numOfClusters, dtype = float)
        while(z < numOfClusters):
            y = 0
            add = 0
            total = 0
            while(y < 783):
                if(trainData[0][y] == centerCluster):
                    add = add + 1
                total =  total + 1
                y = y + 1
            z = z + 1
            j = add / total

            if(z < numOfClusters):
                jacarArray[z] = j

        minIndex = np.argmin(jacarArray)

        if(minIndex == 0):
            clusters1.append(trainData[x])

        if(minIndex == 1):
            clusters2.append(trainData[x])

        if(minIndex == 2):
            clusters3.append(trainData[x])

        if(minIndex == 3):
            clusters4.append(trainData[x])

        if(minIndex == 4):
            clusters5.append(trainData[x])

        if(minIndex == 5):
            clusters6.append(trainData[x])

        if(minIndex == 6):
            clusters7.append(trainData[x])

        if(minIndex == 7):
            clusters8.append(trainData[x])

        if(minIndex == 8):
            clusters9.append(trainData[x])

        if(minIndex == 9):
            clusters10.append(trainData[x])

        z = 0

    clustersWithData = []
    clustersWithData.append(clusters1)
    clustersWithData.append(clusters2)
    clustersWithData.append(clusters3)
    clustersWithData.append(clusters4)
    clustersWithData.append(clusters5)
    clustersWithData.append(clusters6)
    clustersWithData.append(clusters7)
    clustersWithData.append(clusters8)
    clustersWithData.append(clusters9)
    clustersWithData.append(clusters10)

    return  clustersWithData
#Number Of Clusters = K
#randomly pick 10 datapoints as clusters
def randomCenterClusters(npTrainData, numOfClusters):
    centerClusters = []
    center = []

    rows = len(npTrainData)
    #print(rows)
    x = 0
    while(x < numOfClusters):
        indexVal = randint(1, rows - 1)
        #print(indexVal)
        #print(npTrainData[indexVal])

        center.append(npTrainData[indexVal])
        centerClusters.append(center)

        center = []
        x = x + 1

    return np.array(centerClusters)


trainData = readData("./Task2Dataset/data.csv")


npTrainData = np.array(trainData)

numOfClusters = 10  # k = 10
npCenterClusters = randomCenterClusters(npTrainData, numOfClusters)

#print(npCenterClusters)
#print(len(npCenterClusters))

#Using eucledian distance
print("Begining of Eucledian Distance")
clustersWithData, clusters1 = kMeansAlgorithm(numOfClusters, npTrainData, npCenterClusters, 1)
print("Cluster with Data")
print(clustersWithData)
print("Centers")
print(clusters1)
print("End of Eucledian Distance")

print("Begining of Cosine Similarity")
#Using Cosine similarity
clustersWithData, clusters1 = kMeansAlgorithm(numOfClusters, npTrainData, npCenterClusters, 2)
print("Cluster with Data")
print(clustersWithData)
print("Centers")
print(clusters1)
print("End of Cosine Similarity")

print("Beginning Of Jaccard")
#Using jaccard
clustersWithData, clusters1 = kMeansAlgorithm(numOfClusters, npTrainData, npCenterClusters, 3)
print("Cluster with Data")
print(clustersWithData)
print("Centers")
print(clusters1)
print("End of Jaccard")
#print(trainData)
