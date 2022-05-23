import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.cluster import KMeans 

dataset= pd.read_csv('datasrc.csv')
X=dataset.iloc[:,lambda df: [2, 5]].values


kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter=300, n_init=10, random_state=0)

y_kmeans= kmeans.fit_predict(X)


plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=150, color='red', label='cụm 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=150, color='blue', label='cụm 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=150, color='green', label='cụm 3')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:, 1],
            s=100, c='yellow', label='Center')

plt.title('dữ liệu trường tiểu học')
plt.xlabel('trung bình đọc')
plt.ylabel('trung bình toán')
plt.legend(loc='upper right')
plt.rcParams["figure.figsize"]=(20,8)

plt.show()
