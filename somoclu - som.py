import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import somoclu
import pandas as pd


dataset = pd.read_csv('creditcard.csv')
COUNT = 60000
dataset2 = dataset.head(COUNT)
X = dataset2.iloc[:, :-1].values
y = dataset2.iloc[:, -1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
data = sc.fit_transform(X)

colors = ['red' if l == 0 else 'black' for l in dataset['Class']]

n_rows, n_columns = 40, 40
som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)
som.train(data)

from sklearn.cluster import DBSCAN
algorithm = DBSCAN()
som.cluster(algorithm=algorithm)

list = []
for i in range(COUNT):
    list.append(som.clusters[som.bmus[i, 1], som.bmus[i, 0]])

mylist = [-x for x in list]

from sklearn import metrics
score = metrics.f1_score(mylist, y)
print(score)


som.view_umatrix(bestmatches=True, bestmatchcolors=colors )

