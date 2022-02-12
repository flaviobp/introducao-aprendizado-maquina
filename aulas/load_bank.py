#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#carga e preparacao dos dados
df = pd.read_csv('data/tab_bank.csv', sep=';')

y = df['y']
y = np.array(y == 'yes').astype('uint8')

df = df.drop(['y'], axis=1)
df = pd.get_dummies(df)
X = np.array(df)

#split dos dados train/test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    stratify=y,
                                                    random_state=420)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.save('bank_X_train.npy', X_train)
np.save('bank_X_test,npy', X_test)
np.save('bank_y_train.npy', y_train)
np.save('bank_y_test,npy', y_test)


### Treinamento com CV
skf = StratifiedKFold(n_splits=5, random_state=420)

acc_dt = []

for tr_idx, vl_idx in skf.split(X_train, y_train):
    X_train_f, X_valid_f = X_train[tr_idx], X_train[vl_idx]
    y_train_f, y_valid_f = y_train[tr_idx], y_train[vl_idx]

    clf = LogisticRegression(solver='liblinear') #DecisionTreeClassifier('gini')
    clf.fit(X_train_f, y_train_f)
    y_pred_f = clf.predict(X_valid_f)

    acc_dt.append(accuracy_score(y_valid_f, y_pred_f))

    tn, fp, fn, tp = confusion_matrix(y_valid_f, y_pred_f).ravel()

    print('TN: {0}, fp: {1}, fn: {2}, TP: {3}'.format(tn, fp, fn, tp))

print(np.mean(acc_dt))
