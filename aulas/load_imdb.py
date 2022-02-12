#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_csv('data/txt_imdb.csv')
y = np.array(df['y'])

cv = CountVectorizer(stop_words='english', max_features=100)
tf = TfidfVectorizer(stop_words='english', max_features=100)

X_cv = cv.fit_transform(df['X'])
X_tf = tf.fit_transform(df['X'])

cv.get_feature_names()[:100]
