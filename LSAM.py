import numpy as np1
import pandas as pd
from click._compat import raw_input
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def LSAM(X):
    b=np1.mean(X)
    return b


#Read .csv file
df = pd.read_csv('4.StockMarket.csv', header=0)
df2= list(df)
len2=len(df2)
A2=[0]*len2
for i in range(1,len2):
    x=df[df2[i]]
    A2[i]=LSAM(list(x))   #Call to reduction
#print(A2)

finalReduction=[]
for i in range(1,len2):
    finalReduction=finalReduction+[A2[i]]
finalReduction=np1.reshape(finalReduction,[-1,1])
print(finalReduction)
plt.plot( len(finalReduction) * [1],finalReduction, "x")
plt.show()


def optimalCluster(finalReduction):
    sse = {}
    min1 = int(raw_input("Enter minimum number of clusters:"))
    max1 = int(raw_input("Enter maximum number of clusters:"))
    for k in range(min1, max1):
        kmeans = KMeans(n_clusters=k, max_iter=10).fit(finalReduction)
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    return sse

def Clustering(finalReduction,cluster):
    kmeans = KMeans(n_clusters=cluster)
    kmeans.fit(finalReduction)
    labels = kmeans.labels_
    return labels



#Selecting Optimal K value using Sum Square Approach
sse=optimalCluster(finalReduction)
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
#print(list(sse.keys()))
#print(list(sse.values()))


#Apply Clustering
cluster = int(raw_input("Enter optimal number of clusters:"))  # Selecting no.of clusters from graph
finalLabels=Clustering(finalReduction,cluster)
print(finalLabels)