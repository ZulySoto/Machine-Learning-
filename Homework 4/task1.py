import csv
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statistics import mean

# Read data from csv file
def readData(filename):
    file = open(filename)
    reader = pd.read_csv(file)

    trainData = []
    tempArray = []  # Will hold the x and y values
    counter = 0
    while(counter < 10):
        tempArray.append(reader.values[counter][1])
        tempArray.append(reader.values[counter][2])
        trainData.append(tempArray)

        tempArray = []
        counter = counter + 1
    return trainData

#Does the KMeans algorithm
def kMeansAlgorithm(trainData, centerClusters, typeOfDistance):

    stopSignal = 1

    count = 0
    while(stopSignal != 0):
        previousCenterClusters = centerClusters
        clustersWithData = setDatatoCentroid(2, np.array(centerClusters), np.array(trainData), typeOfDistance)

        centerClusters = newCenterClusters(np.array(clustersWithData, dtype=object))

        if(count == 0):
            print("Center Clusters After 1 Interation")
            print(centerClusters)
            count = count + 1
        #Checks if the clusters have changed
        stop = np.subtract(previousCenterClusters, centerClusters)
        if(np.all(stop == 0)):
           stopSignal = 0

    return clustersWithData, centerClusters

#Gets the new centers for the clusters
def newCenterClusters(clustersWithData):
    newCenter1 = []
    newCenter2 = []

    clusterLength1 = len(clustersWithData[0])
    clusterLength2 = len(clustersWithData[1])

    #Cluster 1
    x = 0
    additionX = 0
    additionY = 0
    while(x < clusterLength1):
        #print(clustersWithData[0][x][0])
        additionX = additionX + clustersWithData[0][x][0]
        additionY = additionY + clustersWithData[0][x][1]
        x = x + 1

    newCenterX = additionX/clusterLength1
    newCenterY = additionY/clusterLength1

    newCenter1 = [newCenterX, newCenterY]

    #Cluster 2
    x = 0
    additionX = 0
    additionY = 0
    while(x < clusterLength2):
        #print(clustersWithData[0][x][0])
        additionX = additionX + clustersWithData[1][x][0]
        additionY = additionY + clustersWithData[1][x][1]
        x = x + 1

    newCenterX = additionX/clusterLength2
    newCenterY = additionY/clusterLength2

    newCenter2 = [newCenterX, newCenterY]

    centerClusters = [newCenter1, newCenter2]

    return centerClusters

def setDatatoCentroid(numOfCentroids, centerClusters, trainData, typeOfDistance):
    #Will Hold the new points for the centers of the clusters
    centroid1 = []
    centroid2 = []

    #manhatan distance
    if(typeOfDistance == 0):
        #Loops  for
        for x in trainData:
            subXC1 = np.subtract(trainData[x][0][0], centerClusters[0][0])
            absXC1 = abs(subXC1)
            subYC1 = np.subtract(trainData[x][0][1], centerClusters[0][1])
            absYC1 = abs(subYC1)
            addC1 = absXC1 + absYC1

            subXC2 = np.subtract(trainData[x][0][0], centerClusters[1][0])
            absXC2 = abs(subXC2)
            subYC2 = np.subtract(trainData[x][0][1], centerClusters[1][1])
            absYC2 = abs(subYC2)
            addC2 = absXC2 + absYC2

            minimumDistance = min(addC1, addC2)

            if(minimumDistance == addC1):
                centroid1.append(trainData[x][0])
            if(minimumDistance == addC2):
                centroid2.append(trainData[x][0])

    #Eucledian Distance
    if(typeOfDistance == 1):
        for x in trainData:
            subXC1 = np.subtract(trainData[x][0][0], centerClusters[0][0])
            squaredXC1 = subXC1 * subXC1
            subYC1 = np.subtract(trainData[x][0][1], centerClusters[0][1])
            squaredYC1 = subYC1 * subYC1
            addC1 = squaredXC1 + squaredYC1
            #(addC1)
            squaredRootC1 = math.sqrt(addC1)


            subXC2 = np.subtract(trainData[x][0][0], centerClusters[1][0])
            squaredXC2 = subXC2 * subXC2
            subYC2 = np.subtract(trainData[x][0][1], centerClusters[1][1])
            squaredYC2 = subYC2 * subYC2
            addC2 = squaredXC2 + squaredYC2
            squaredRootC2 = math.sqrt(addC2)

            minimumDistance = min(squaredRootC1, squaredRootC2)

            if(minimumDistance == squaredRootC1):
                centroid1.append(trainData[x][0])
            if(minimumDistance == squaredRootC2):
                centroid2.append(trainData[x][0])


    clustersWithData = [centroid1, centroid2]

    return clustersWithData
#First value is the x and second value is the y
trainData = readData('Task1DataSet.csv')


#Centroid Initialization
centerClusters = [[4, 6], [5, 4]]
clusters1, centerCluster1 = kMeansAlgorithm(trainData, np.array(centerClusters), 0)
clusters2, centerCluster2 = kMeansAlgorithm(trainData, np.array(centerClusters), 1)

centerClusters = [[3, 3], [8,3]]
clusters3, centerCluster3 = kMeansAlgorithm(trainData, np.array(centerClusters), 0)

centerClusters = [[3, 2], [4, 8]]
clusters4, centerCluster4 = kMeansAlgorithm(trainData, np.array(centerClusters), 0)

#Prints out all the clusters of data
print(clusters1)
print(clusters2)
print(clusters3)
print(clusters4)

print("-------------------------------------------------")
print(centerCluster1)
print(centerCluster2)
print(centerCluster3)
print(centerCluster4)
#K Means Algorithm
# Specify the number of k of clusters to assign (pick k as initial centroids)
# Randomly initialize k centroids
# repeat
#    assign each point to its closest centroid
#    compute the new centroid using the mean of each cluster using distances
# until the centroid positions do not change
