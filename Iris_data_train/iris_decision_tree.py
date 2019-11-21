#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 22:15:51 2019

@author: cynthia
"""


import pandas as pd

df=pd.read_csv('/Users/cynthia/Downloads/iris.data',header=None)
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'class']
df['class'] = df['class'].apply(lambda x: x.split('-')[1]) 

# 首先对数据进行切分，即分出数据集和测试集
from sklearn.model_selection import train_test_split

all_inputs = df[['sepal_len', 'sepal_width',
                             'petal_len', 'petal_width']].values
all_classes = df['class'].values

(X_train,
 X_test,
 Y_train,
 Y_test) = train_test_split(all_inputs, all_classes, train_size=0.8, random_state=1)
 

# 使用决策树算法进行训练
from sklearn.tree import DecisionTreeClassifier

# 定义一个决策树对象
decision_tree_classifier = DecisionTreeClassifier()

# 训练模型
model = decision_tree_classifier.fit(X_train, Y_train)

# 所得模型的准确性
print(decision_tree_classifier.score(X_test, Y_test))


print(X_test[0:3])
print(Y_test[0:3])
model.predict(X_test[0:3])
