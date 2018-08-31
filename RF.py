from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler

FILE_NAME = 'creditcard.csv'
LABEL_STR = 'Class'
splitRate = 0.8


import pandas as pd
import numpy as np

#df = pd.read_csv("elektrownie.csv")
#df_test = pd.read_csv("test.csv")
#submissions = pd.read_csv("sample_submission.csv")

df_inputs = pd.read_csv(FILE_NAME)


X = df_inputs.drop([LABEL_STR], axis=1)
y = df_inputs[LABEL_STR]

train_size = int(splitRate * len(X))
(raw_X_train, y_train) = (X[:train_size], y[:train_size])
(raw_X_test, y_test) = (X[train_size:], y[train_size:])


min_max=MinMaxScaler()

X_train = min_max.fit_transform(raw_X_train)
X_test = min_max.fit_transform(raw_X_test)

tscv = TimeSeriesSplit(n_splits=3)
my_cv = tscv.split(X_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0, verbose = 10, n_estimators = 100)


# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

grid_search = GridSearchCV(clf, param_grid=param_grid, verbose=2, cv=my_cv)
grid_search.fit(X, y)

y_pred = clf.predict(X_test)

from sklearn import metrics
print(len(y_test.values))
print(len(y_pred))
score = metrics.f1_score(y_test.values, y_pred)
score
#y  

#clf = RandomForestClassifier(max_depth=2, random_state=0, verbose = 10, n_estimators = 100)
#clf.fit(X_train, y_train)