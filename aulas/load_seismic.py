#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data, _ = loadarff('data/tab_seismic_bumps.arff')

df = pd.DataFrame.from_records(data)

df['seismic'] = df['seismic'].str.decode('utf-8')
df['seismoacoustic'] = df['seismoacoustic'].str.decode('utf-8')
df['shift'] = df['shift'].str.decode('utf-8')
df['ghazard'] = df['ghazard'].str.decode('utf-8')
df['class'] = df['class'].str.decode('utf-8')

y = np.array((df['class'] == '1').astype('uint8'))
X = np.array(pd.get_dummies(df.drop('class', axis=1)))

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=420)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


skf = StratifiedKFold(n_splits=5, random_state=420)

acc_dt = []

for tr_idx, vl_idx in skf.split(X_train, y_train):
    X_train_f, X_valid_f = X_train[tr_idx], X_train[vl_idx]
    y_train_f, y_valid_f = y_train[tr_idx], y_train[vl_idx]

    clf = DecisionTreeClassifier('gini')
    clf.fit(X_train_f, y_train_f)
    y_pred_f = clf.predict(X_valid_f)

    acc_dt.append(accuracy_score(y_valid_f, y_pred_f))

    tn, fp, fn, tp = confusion_matrix(y_valid_f, y_pred_f).ravel()

    print('TN: {0}, fp: {1}, fn: {2}, TP: {3}'.format(tn, fp, fn, tp))

print(np.mean(acc_dt))
