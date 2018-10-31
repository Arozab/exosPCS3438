from sklearn.linear_model import Lasso

import csv
import random
import math
import numpy as np
import metrics

def loadDataset(filename, split):

    with open(filename, 'r') as csvfile:

        lines = csv.reader(csvfile)
        header = next(lines)
        dataset = list(lines)
        trainSize = int(split*len(dataset))

        trainset = dataset[:trainSize]
        testset = dataset[trainSize:]
        print(len(trainset))
        print(len(testset))
        return [trainset, testset]

data_train,data_test = loadDataset('reg01.csv',0.9)

X_train = [[float(x) for x in t] for t in data_train]
y_train = [float(t[-1]) for t in data_train]
X_test = [[float(x) for x in t]  for t in data_test]
y_test = [float(t[-1]) for t in data_test]
print(y_test)
print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

las = Lasso(alpha=1, normalize=True)
las.fit(X_train, y_train)
las.coef_
preds = las.predict(X_test)
print('RMSE (Lasso reg.) =', np.sqrt(metrics.mean_squared_error(y_test, preds)))
