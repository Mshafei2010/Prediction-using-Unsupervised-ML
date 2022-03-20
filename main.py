import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import seaborn as sns

# IMPORTING Data
iris= datasets.load_iris()
iris_df= pd.DataFrame(iris.data , columns=iris.feature_names)
print("data imported")
print(iris_df.head())


#DETERMINING THE OPTIMUM NUMBER OF CLUSTERS USING THE ELBOW METHOD
x=iris_df.iloc[:,[0,1,2,3]].values
wcss=[]
for i in range (1,11):
    kmeans = KMeans(n_clusters =i, init="k-means++",max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# CREATING THE KMEANS CLASSIFIER
kmeans = KMeans(n_clusters =i, init="k-means++",max_iter=300,n_init=10,random_state=0)
y_kmeans = kmeans.fit_predict(x)

# PLOTING THE CLUSTERS
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label='iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='orange',label='iris-versicolour')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='iris-virginica')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='purple',label='Centroids')
plt.legend()

# 3d plot

fig = plt.figure(figsize = (15,15))
ax = fig.add_subplot(111, projection='3d')
plt.scatter(x[y_kmeans==0, 0],x[y_kmeans==0, 1],s=50,c='red',label='iris-setsosa')
plt.scatter(x[y_kmeans==1, 0],x[y_kmeans==1, 1],s=50,c='orange',label='iris-versicolour')
plt.scatter(x[y_kmeans==2, 0],x[y_kmeans==2, 1],s=50,c='green',label='iris-virginica')

# plotting centriods
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,c='purple',label='Centriod')

plt.show()
