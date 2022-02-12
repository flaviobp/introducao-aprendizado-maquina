#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_csv('data/txt_twitter_airline.csv')

lenc = LabelEncoder()
y = lenc.fit_transform(df['airline_sentiment'])

cv = CountVectorizer(stop_words='english', max_features=100)
tf = TfidfVectorizer(stop_words='english', max_features=100)

#tfidf(t, d) = tf(t, d) * idf(t)
#
#idf(t) = log [ n / (df(t) + 1)]

X_cv = cv.fit_transform(df['text'])
X_tf = tf.fit_transform(df['text'])
